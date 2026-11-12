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

from .r1_jet_env_cfg import WHOLE_BODY_JOINTS, R1JetEnvCfg

BOOSTER_LENGTH = 0.30
FLAME_LENGTH = 0.60  # at full throttle


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
            for k in ["pos_coarse", "pos_fine", "progress", "action_bound", "upright", "lin_vel", "ang_vel", "yaw_rate", "action_rate", "throttle", "joint_dev", "termination"]
        }
        self.set_debug_vis(self.cfg.debug_vis)

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # point the camera (world coords) at env 0's flight volume; DirectRLEnv copies cfg.viewer
        # into the video recorder after this, so this also frames the --video / W&B clips
        o = self._terrain.env_origins[0].tolist()
        self.cfg.viewer.eye = (o[0] + 6.5, o[1] + 6.5, o[2] + 4.0)
        self.cfg.viewer.lookat = (o[0], o[1], o[2] + 2.2)

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
        too_far = torch.linalg.norm(rel[:, :2], dim=1) > self.scene.cfg.env_spacing / 2.0
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
        if debug_vis:
            if not hasattr(self, "_goal_vis"):
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
                self._booster_vis = VisualizationMarkers(
                    VisualizationMarkersCfg(
                        prim_path="/Visuals/R1Jet/booster",
                        markers={
                            "booster": sim_utils.CylinderCfg(
                                radius=0.055,
                                height=BOOSTER_LENGTH,
                                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.28), metallic=0.8),
                            )
                        },
                    )
                )
                self._flame_vis = VisualizationMarkers(
                    VisualizationMarkersCfg(
                        prim_path="/Visuals/R1Jet/flame",
                        markers={
                            "flame": sim_utils.ConeCfg(
                                radius=0.05,
                                height=FLAME_LENGTH,
                                visual_material=sim_utils.PreviewSurfaceCfg(
                                    diffuse_color=(1.0, 0.5, 0.1), emissive_color=(1.0, 0.45, 0.05)
                                ),
                            )
                        },
                    )
                )
            for v in (self._goal_vis, self._booster_vis, self._flame_vis):
                v.set_visibility(True)
        else:
            for name in ("_goal_vis", "_booster_vis", "_flame_vis"):
                if hasattr(self, name):
                    getattr(self, name).set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self._robot.is_initialized:
            return
        self._goal_vis.visualize(self._goal_pos_w)

        link_pos, link_quat, _ = self._booster_body_state()
        n = self.num_envs
        q_link = link_quat.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4)
        ex = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(2 * n, 3)
        q_booster = quat_mul(q_link, self._booster_quats_b())
        nozzle = link_pos.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 3) + quat_apply(q_link, self._mounts_b.repeat(n, 1))
        thrust_dir = quat_apply(q_booster, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(2 * n, 3))
        # casing sits above the nozzle along the thrust axis
        self._booster_vis.visualize(nozzle + thrust_dir * (BOOSTER_LENGTH / 2), q_booster)
        # flame points out of the nozzle, opposite the thrust; cone apex (+z) flipped to face away
        q_flip = quat_from_angle_axis(torch.full((2 * n,), math.pi, device=self.device), ex)
        throttle = self._throttle.reshape(-1).clamp(min=0.02)
        scales = torch.stack([0.6 + 0.6 * throttle, 0.6 + 0.6 * throttle, throttle], dim=-1)
        self._flame_vis.visualize(
            nozzle - thrust_dir * (FLAME_LENGTH * throttle / 2).unsqueeze(-1), quat_mul(q_booster, q_flip), scales
        )
