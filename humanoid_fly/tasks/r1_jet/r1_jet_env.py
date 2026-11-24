"""R1 jet-flight environment (direct workflow, modelled on Isaac Lab's quadcopter task)."""

from __future__ import annotations

import math

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_angle_axis, quat_from_euler_xyz, quat_mul, subtract_frame_transforms

from .. import visuals
from .r1_jet_env_cfg import WHOLE_BODY_JOINTS, R1JetEnvCfg

BOOSTER_LENGTH = 0.30  # casing, visual only
BOOSTER_RADIUS = 0.055
FLAME_LENGTH = 0.60  # at full throttle
FLAME_RADIUS = 0.05


class R1JetEnv(DirectRLEnv):
    cfg: R1JetEnvCfg

    def __init__(self, cfg: R1JetEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device

        self._actions = torch.zeros(n, self.cfg.action_space, device=dev)
        self._prev_actions = torch.zeros_like(self._actions)
        self._raw_actions = torch.zeros_like(self._actions)

        # booster state (with lag): throttle in [0, 1], gimbal angles [rad] as (pitch_L, pitch_R, roll_L, roll_R)
        self._throttle_cmd = torch.zeros(n, 2, device=dev)
        self._gimbal_cmd = torch.zeros(n, 4, device=dev)
        self._throttle = torch.zeros(n, 2, device=dev)
        self._gimbal = torch.zeros(n, 4, device=dev)
        self._mounts_b = torch.tensor(self.cfg.booster_mounts, device=dev)  # (2, 3)

        self._body_id = self._robot.find_bodies(self.cfg.booster_body)[0]
        assert len(self._body_id) == 1, f"booster body {self.cfg.booster_body} not found"
        if self.cfg.control_joints:
            self._ctrl_joint_ids = self._robot.find_joints(WHOLE_BODY_JOINTS)[0]
            assert len(self._ctrl_joint_ids) == self.cfg.action_space - 6, (
                f"expected {self.cfg.action_space - 6} controlled joints, found {len(self._ctrl_joint_ids)}"
            )
        assert self._robot.num_joints == (self.cfg.observation_space - 19 - self.cfg.action_space) // 2

        self._q_cant_b, self._thrust_axis_b = self._compute_cant()
        vis_axis = torch.tensor(self._vis_axis_b(), device=dev)
        err = math.degrees(math.acos(max(-1.0, min(1.0, float(torch.dot(vis_axis, self._thrust_axis_b))))))
        if self._vis_pivots and err > 1.0:
            true_deg = math.degrees(math.atan2(self._thrust_axis_b[0].item(), self._thrust_axis_b[2].item()))
            print(f"[R1Jet] WARNING: booster casings are drawn {err:.1f} deg off the real neutral thrust axis; "
                  f"set booster_vis_cant_deg = {true_deg:.1f} in R1JetEnvCfg")

        mass = wp.to_torch(self._robot.root_view.get_masses())[0].sum().item()
        g = torch.tensor(self.sim.cfg.gravity).norm().item()
        self._robot_weight = mass * g
        self._max_thrust = self.cfg.thrust_to_weight * self._robot_weight / 2.0  # per booster
        self._hover_throttle = 1.0 / self.cfg.thrust_to_weight
        print(f"[R1Jet] robot mass {mass:.1f} kg, max thrust per booster {self._max_thrust:.0f} N")

        self._goal_pos_w = torch.zeros(n, 3, device=dev)
        self._prev_dist = torch.zeros(n, device=dev)
        self._goal_resample_steps = max(1, int(self.cfg.goal_resample_time_s / self.step_dt))

        self._episode_sums = {
            k: torch.zeros(n, device=dev)
            for k in ["pos_coarse", "pos_fine", "progress", "action_bound", "upright", "lin_vel", "ang_vel", "yaw_rate", "action_rate", "throttle", "joint_dev", "hold_ang_vel", "hold_lin_vel", "hold_action_rate", "termination"]
        }
        self.set_debug_vis(self.cfg.debug_vis)

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # jetpack is authored on env 0's robot before cloning, as children of the torso link, so every env gets it
        # and it moves rigidly with the back (no render lag)
        self._vis_pivots = self._attach_jetpack_visuals() if (self.cfg.debug_vis and self.cfg.thruster_vis == "attached") else []
        self.scene.clone_environments(copy_from_source=False)
        self._live_flames = (
            visuals.LiveFlames(self.cfg.robot.prim_path, self._vis_pivots, self.scene.cfg.num_envs)
            if self._vis_pivots and self.cfg.live_flames else None
        )
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # training clips / W&B: hero robot (env 0) in front, the rest of the grid behind it.
        # DirectRLEnv copies cfg.viewer into the video recorder after this.
        self.cfg.viewer.eye, self.cfg.viewer.lookat = visuals.grid_camera(
            self._terrain.env_origins, self.cfg.camera_back, self.cfg.camera_height, self.cfg.camera_ahead,
            self.cfg.camera_look_height,
        )

    def _vis_axis_b(self) -> tuple[float, float, float]:
        """Neutral booster axis (thrust direction) used for the casings, torso frame."""
        c = math.radians(self.cfg.booster_vis_cant_deg)
        return (math.sin(c), 0.0, math.cos(c))

    def _attach_jetpack_visuals(self) -> list[str]:
        """Two booster casings and a pack plate on the torso link, plus a flame out of the bottom of each casing.

        The physics applies each booster's thrust at its mount point (shoulder-blade height) along the booster axis.
        The casing runs from that point *down* the back along the same axis, and the flame leaves its bottom end: the
        force acts along the same line, so drawing the exit lower changes nothing physically, and the pack sits on
        the back instead of sticking up past the shoulders. Returns the flame pivots relative to the robot root.
        """
        roots = visuals.robot_roots(self.cfg.robot.prim_path)
        if not roots:
            print(f"[R1Jet] no robot prim at {self.cfg.robot.prim_path}: no jetpack visuals")
            return []
        axis = self._vis_axis_b()
        exhaust = tuple(-c for c in axis)
        back = (-self.cfg.booster_vis_back_offset, 0.0, 0.0)
        thr = self.cfg.attached_flame_throttle
        rel = []
        for root in roots:  # normally just env 0 (cloned afterwards)
            links = visuals.find_links(root, [self.cfg.booster_body])
            if links is None:
                print("[R1Jet] can't attach jetpack visuals: none drawn")
                return []
            link = str(links[self.cfg.booster_body].GetPath())
            rel, centers = [], []
            for i, mount in enumerate(self.cfg.booster_mounts):
                top = visuals.vadd(mount, back)
                center = visuals.vadd(top, axis, -BOOSTER_LENGTH / 2)
                centers.append(center)
                visuals.add_cylinder(link, f"jetpack_booster_{i}", center, axis, BOOSTER_RADIUS, BOOSTER_LENGTH,
                                     color=(0.25, 0.25, 0.28))
                exit_pt = visuals.vadd(top, axis, -BOOSTER_LENGTH)
                pivot = visuals.add_flame(
                    link, f"{visuals.FLAME_PIVOT}_{i}", exit_pt, exhaust, radius=FLAME_RADIUS,
                    length=FLAME_LENGTH * thr, width=0.6 + 0.6 * thr,
                    color=(1.0, 0.5, 0.1), emissive=(1.0, 0.45, 0.05),
                )
                rel.append(pivot[len(root) + 1:])
            # pack plate between the casings, slightly in front of them (towards the back of the robot)
            mid = tuple(sum(c[k] for c in centers) / len(centers) for k in range(3))
            span = abs(self.cfg.booster_mounts[0][1] - self.cfg.booster_mounts[1][1])
            visuals.add_box(link, "jetpack_pack", visuals.vadd(mid, (1.0, 0.0, 0.0), 0.02),
                            (0.04, span + 0.06, BOOSTER_LENGTH * 0.9), axis, color=(0.18, 0.18, 0.2))
        print(f"[R1Jet] jetpack attached to {self.cfg.booster_body} ({len(roots)} robot prim(s) before cloning)")
        return rel

    # ------------------------------------------------------------------ actions
    def _pre_physics_step(self, actions: torch.Tensor):
        self._prev_actions[:] = self._actions
        self._raw_actions = actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._throttle_cmd = (self._actions[:, 0:2] + 1.0) / 2.0
        self._gimbal_cmd = self._actions[:, 2:6] * self.cfg.max_gimbal

        if self.cfg.control_joints:
            targets = self._robot.data.default_joint_pos.torch.clone()
            targets[:, self._ctrl_joint_ids] += self.cfg.joint_action_scale * self._actions[:, 6:]
        else:
            targets = self._robot.data.default_joint_pos.torch
        self._robot.set_joint_position_target_index(target=targets)

        # new goal every goal_resample_time_s
        resample = (self.episode_length_buf % self._goal_resample_steps == 0) & (self.episode_length_buf > 0)
        ids = resample.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            self._sample_goals(ids)
            self._prev_dist[ids] = torch.linalg.norm(self._goal_pos_w[ids] - self._robot.data.root_pos_w.torch[ids], dim=1)

    def _apply_action(self):
        # first-order lag on throttle and gimbals, integrated at the physics rate
        dt = self.physics_dt
        self._throttle += (self._throttle_cmd - self._throttle) * min(1.0, dt / self.cfg.throttle_time_constant)
        self._gimbal += (self._gimbal_cmd - self._gimbal) * min(1.0, dt / self.cfg.gimbal_time_constant)

        link_pos, link_quat, com_pos = self._booster_body_state()
        dir_w = self._thrust_dirs_w(link_quat)  # (n, 2, 3)
        force_w = dir_w * (self._throttle * self._max_thrust).unsqueeze(-1)
        nozzle_w = link_pos.unsqueeze(1) + quat_apply(
            link_quat.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4), self._mounts_b.repeat(self.num_envs, 1)
        ).view(-1, 2, 3)
        torque_w = torch.cross(nozzle_w - com_pos.unsqueeze(1), force_w, dim=-1)
        self._robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force_w.sum(1, keepdim=True),
            torques=torque_w.sum(1, keepdim=True),
            body_ids=self._body_id,
            is_global=True,
        )

    def _booster_body_state(self):
        d = self._robot.data
        i = self._body_id[0]
        return d.body_link_pos_w.torch[:, i], d.body_link_quat_w.torch[:, i], d.body_com_pos_w.torch[:, i]

    def _compute_cant(self):
        """Neutral booster axis in the torso frame: the line from the whole-body CoM (default pose) through the
        nozzles' midpoint, so zero gimbal produces no pitch torque. Returns (quat (4,), axis (3,))."""
        ey = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        if not self.cfg.cant_through_com:
            return quat_from_angle_axis(torch.zeros(1, device=self.device), ey[None])[0], torch.tensor(
                [0.0, 0.0, 1.0], device=self.device
            )
        d = self._robot.data
        m = wp.to_torch(self._robot.root_view.get_masses())[0].to(self.device)
        com_w = (d.body_com_pos_w.torch[0] * m[:, None]).sum(0) / m.sum()
        i = self._body_id[0]
        com_b = quat_apply_inverse(d.body_link_quat_w.torch[0:1, i], (com_w - d.body_link_pos_w.torch[0, i])[None])[0]
        r = self._mounts_b.mean(0) - com_b
        cant = math.atan2(r[0].item(), r[2].item())  # rotation about torso y taking +z onto (r_x, 0, r_z)
        print(
            f"[R1Jet] CoM in torso frame {[round(x, 3) for x in com_b.tolist()]}, nozzles {r[2].item():.2f} m above it;"
            f" booster cant {math.degrees(cant):.1f} deg -> hovers leaning ~{abs(math.degrees(cant)):.0f} deg forward"
        )
        q = quat_from_angle_axis(torch.tensor([cant], device=self.device), ey[None])[0]
        return q, torch.tensor([math.sin(cant), 0.0, math.cos(cant)], device=self.device)

    def _booster_quats_b(self) -> torch.Tensor:
        """Booster orientations in the torso frame, (n*2, 4): cant, then gimbal Rx(roll) * Ry(pitch)."""
        n = self.num_envs
        ex = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(2 * n, 3)
        ey = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(2 * n, 3)
        q_g = quat_mul(
            quat_from_angle_axis(self._gimbal[:, 2:4].reshape(-1), ex),
            quat_from_angle_axis(self._gimbal[:, 0:2].reshape(-1), ey),
        )
        return quat_mul(self._q_cant_b.expand(2 * n, 4), q_g)

    def _thrust_dirs_b(self) -> torch.Tensor:
        """Unit thrust directions in the torso frame, (n, 2, 3)."""
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(2 * self.num_envs, 3)
        return quat_apply(self._booster_quats_b(), ez).view(-1, 2, 3)

    def _thrust_dirs_w(self, link_quat: torch.Tensor) -> torch.Tensor:
        q = link_quat.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4)
        return quat_apply(q, self._thrust_dirs_b().reshape(-1, 3)).view(-1, 2, 3)

    # ------------------------------------------------------------------ MDP
    def _get_observations(self) -> dict:
        d = self._robot.data
        goal_b, _ = subtract_frame_transforms(d.root_pos_w.torch, d.root_quat_w.torch, self._goal_pos_w)
        height = d.root_pos_w.torch[:, 2:3] - self._terrain.env_origins[:, 2:3]
        obs = torch.cat(
            [
                d.root_lin_vel_b.torch,
                d.root_ang_vel_b.torch,
                d.projected_gravity_b.torch,
                goal_b,
                height,
                self._throttle * 2.0 - 1.0,
                self._gimbal / self.cfg.max_gimbal,
                d.joint_pos.torch - d.default_joint_pos.torch,
                d.joint_vel.torch * 0.1,
                self._actions,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        d = self._robot.data
        dist = torch.linalg.norm(self._goal_pos_w - d.root_pos_w.torch, dim=1)
        # tilt of the neutral thrust axis from vertical (0 = hover pose, 1 = sideways, 2 = upside down)
        _, torso_quat, _ = self._booster_body_state()
        tilt = 1.0 - quat_apply(torso_quat, self._thrust_axis_b.expand(self.num_envs, 3))[:, 2]
        joint_dev = torch.sum(torch.square(d.joint_pos.torch - d.default_joint_pos.torch), dim=1)
        dt = self.step_dt
        hold = torch.exp(-torch.square(dist / self.cfg.hold_radius))
        rewards = {
            "pos_coarse": (1.0 - torch.tanh(dist / 1.5)) * self.cfg.pos_coarse_reward_scale * dt,
            "pos_fine": (1.0 - torch.tanh(dist / 0.3)) * self.cfg.pos_fine_reward_scale * dt,
            "progress": ((self._prev_dist - dist) / dt).clamp(-2.0, 2.0) * self.cfg.progress_reward_scale * dt,
            "action_bound": torch.sum(torch.square(torch.relu(self._raw_actions.abs() - 1.0)), dim=1)
            * self.cfg.action_bound_reward_scale
            * dt,
            "upright": tilt * self.cfg.upright_reward_scale * dt,
            "lin_vel": torch.sum(torch.square(d.root_lin_vel_b.torch), dim=1) * self.cfg.lin_vel_reward_scale * dt,
            "ang_vel": torch.sum(torch.square(d.root_ang_vel_b.torch), dim=1) * self.cfg.ang_vel_reward_scale * dt,
            "yaw_rate": torch.square(d.root_ang_vel_b.torch[:, 2]) * self.cfg.yaw_rate_reward_scale * dt,
            "action_rate": torch.sum(torch.square(self._actions - self._prev_actions), dim=1)
            * self.cfg.action_rate_reward_scale
            * dt,
            "throttle": self._throttle.mean(dim=1) * self.cfg.throttle_reward_scale * dt,
            "joint_dev": joint_dev * self.cfg.joint_dev_reward_scale * dt * float(self.cfg.control_joints),
            # hold at the goal: extra penalties on roll/pitch/yaw rate, drift and jitter that fade in near the goal
            # (gate = 1 at the goal, ~0.37 at hold_radius, ~0 beyond 2x), so they don't slow the flight between goals
            "hold_ang_vel": hold * torch.sum(torch.square(d.root_ang_vel_b.torch), dim=1) * self.cfg.hold_ang_vel_reward_scale * dt,
            "hold_lin_vel": hold * torch.sum(torch.square(d.root_lin_vel_b.torch), dim=1) * self.cfg.hold_lin_vel_reward_scale * dt,
            "hold_action_rate": hold * torch.sum(torch.square(self._actions - self._prev_actions), dim=1)
            * self.cfg.hold_action_rate_reward_scale
            * dt,
            "termination": self.reset_terminated.float() * self.cfg.termination_penalty,
        }
        self._prev_dist[:] = dist
        for k, v in rewards.items():
            self._episode_sums[k] += v
        return torch.sum(torch.stack(list(rewards.values())), dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        d = self._robot.data
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        rel = d.root_pos_w.torch - self._terrain.env_origins
        fell = rel[:, 2] < self.cfg.min_height
        too_high = rel[:, 2] > self.cfg.max_height
        too_far = torch.linalg.norm(rel[:, :2], dim=1) > self.cfg.max_xy_dist
        tilted = -d.projected_gravity_b.torch[:, 2] < self.cfg.max_tilt_cos
        died = fell | too_high | too_far | tilted
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)

        final_dist = torch.linalg.norm(
            self._goal_pos_w[env_ids] - self._robot.data.root_pos_w.torch[env_ids], dim=1
        ).mean()
        self.extras["log"] = {}
        for k in self._episode_sums:
            self.extras["log"]["Episode_Reward/" + k] = torch.mean(self._episode_sums[k][env_ids]) / self.max_episode_length_s
            self._episode_sums[k][env_ids] = 0.0
        self.extras["log"]["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"]["Metrics/final_distance_to_goal"] = final_dist.item()

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # stagger episodes so resets don't all happen on the same step
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        n = len(env_ids)
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._gimbal[env_ids] = 0.0
        self._gimbal_cmd[env_ids] = 0.0

        # robot state: standing on the ground, or tossed into the air already at hover throttle
        root_pose = self._robot.data.default_root_pose.torch[env_ids].clone()
        root_vel = torch.zeros(n, 6, device=self.device)
        root_pose[:, :3] += self._terrain.env_origins[env_ids]
        air = torch.rand(n, device=self.device) < self.cfg.air_start_prob
        yaw = (torch.rand(n, device=self.device) * 2 - 1) * math.pi
        roll = torch.where(air, (torch.rand(n, device=self.device) * 2 - 1) * 0.3, torch.zeros(n, device=self.device))
        pitch = torch.where(air, (torch.rand(n, device=self.device) * 2 - 1) * 0.3, torch.zeros(n, device=self.device))
        root_pose[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
        root_pose[air, 2] += torch.empty(int(air.sum()), device=self.device).uniform_(0.5, 2.5)
        root_vel[air, :3] = torch.empty(int(air.sum()), 3, device=self.device).uniform_(-1.0, 1.0)
        root_vel[air, 3:] = torch.empty(int(air.sum()), 3, device=self.device).uniform_(-0.5, 0.5)
        start_throttle = torch.where(air, self._hover_throttle, 0.0).unsqueeze(-1).expand(-1, 2)
        self._throttle[env_ids] = start_throttle
        self._throttle_cmd[env_ids] = start_throttle

        self._robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self._robot.write_joint_position_to_sim_index(
            position=self._robot.data.default_joint_pos.torch[env_ids], env_ids=env_ids
        )
        self._robot.write_joint_velocity_to_sim_index(
            velocity=self._robot.data.default_joint_vel.torch[env_ids], env_ids=env_ids
        )
        self._sample_goals(env_ids)
        self._prev_dist[env_ids] = torch.linalg.norm(self._goal_pos_w[env_ids] - root_pose[:, :3], dim=1)

    def _sample_goals(self, env_ids: torch.Tensor):
        n = len(env_ids)
        r = self.cfg.goal_xy_range
        self._goal_pos_w[env_ids, :2] = torch.empty(n, 2, device=self.device).uniform_(-r, r)
        self._goal_pos_w[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*self.cfg.goal_z_range)
        self._goal_pos_w[env_ids] += self._terrain.env_origins[env_ids]

    # ------------------------------------------------------------------ visuals
    def _set_debug_vis_impl(self, debug_vis: bool):
        # only the goal is a marker; the jetpack is attached to the torso (see _attach_jetpack_visuals)
        if debug_vis and not hasattr(self, "_goal_vis"):
            self._goal_vis = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/R1Jet/goal",
                    markers={
                        "goal": sim_utils.SphereCfg(
                            radius=0.12,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2)),
                        )
                    },
                )
            )
        if hasattr(self, "_goal_vis"):
            self._goal_vis.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        # runs right before each render
        if not self._robot.is_initialized:
            return
        self._goal_vis.visualize(self._goal_pos_w)
        if self._live_flames is not None:
            thr = self._throttle.clamp(min=0.02)
            exhaust_b = -self._thrust_dirs_b()  # (n, 2, 3) torso frame, includes cant + gimbal
            self._live_flames.update((FLAME_LENGTH * thr).cpu(), (0.6 + 0.6 * thr).cpu(), exhaust_b.cpu())
