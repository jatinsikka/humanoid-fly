"""R1 Iron Man flight: thrusters in the feet and hands, steered by moving the limbs."""

from __future__ import annotations

import math

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    subtract_frame_transforms,
)
from isaaclab.utils.string import resolve_matching_names_values

from .. import visuals
from .r1_ironman_env_cfg import CONTROL_JOINTS, NUM_CONTROL_JOINTS, R1IronManEnvCfg

# flame length at full throttle [m]: feet, hands (visual only)
FLAME_LENGTH = (0.70, 0.70, 0.40, 0.40)
FLAME_RADIUS = 0.045


class R1IronManEnv(DirectRLEnv):
    cfg: R1IronManEnvCfg

    def __init__(self, cfg: R1IronManEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        nt = len(self.cfg.thruster_bodies)

        self._actions = torch.zeros(n, self.cfg.action_space, device=dev)
        self._prev_actions = torch.zeros_like(self._actions)
        self._raw_actions = torch.zeros_like(self._actions)
        self._throttle_cmd = torch.zeros(n, nt, device=dev)
        self._throttle = torch.zeros(n, nt, device=dev)

        self._thruster_body_ids = [self._robot.find_bodies(b, preserve_order=True)[0][0] for b in self.cfg.thruster_bodies]
        self._nozzles_b = torch.tensor(self.cfg.thruster_nozzles, device=dev)  # (nt, 3)
        ex = torch.tensor(self.cfg.thruster_exhaust, device=dev)
        self._exhaust_b = ex / ex.norm(dim=-1, keepdim=True)  # (nt, 3)

        ids, names, scales = resolve_matching_names_values(CONTROL_JOINTS, self._robot.joint_names)
        assert len(ids) == NUM_CONTROL_JOINTS, f"expected {NUM_CONTROL_JOINTS} controlled joints, got {names}"
        self._ctrl_joint_ids = ids
        self._joint_scales = torch.tensor(scales, device=dev)
        assert self._robot.num_joints * 2 + 29 + self.cfg.action_space + 2 * self.cfg.heading_command == self.cfg.observation_space

        mass = wp.to_torch(self._robot.root_view.get_masses())[0].sum().item()
        g = torch.tensor(self.sim.cfg.gravity).norm().item()
        self._robot_weight = mass * g
        self._max_thrust = torch.tensor(self.cfg.thruster_max_twr, device=dev) * self._robot_weight  # (nt,)
        self._hover_throttle = 1.0 / sum(self.cfg.thruster_max_twr)
        print(f"[R1IronMan] robot mass {mass:.1f} kg, max thrust per thruster {self._max_thrust.tolist()} N")
        print(f"[R1IronMan] controlled joints: {names}")

        self._goal_pos_w = torch.zeros(n, 3, device=dev)
        self._target_yaw = torch.zeros(n, device=dev)
        self._prev_dist = torch.zeros(n, device=dev)
        self._goal_resample_steps = max(1, int(self.cfg.goal_resample_time_s / self.step_dt))

        self._episode_sums = {
            k: torch.zeros(n, device=dev)
            for k in [
                "pos_coarse", "pos_fine", "progress", "action_bound", "upright", "lin_vel", "ang_vel",
                "action_rate", "throttle", "joint_dev", "joint_vel", "heading", "yaw_rate", "termination",
            ]
        }
        self.set_debug_vis(self.cfg.debug_vis)

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # flames are authored on env 0's robot before cloning, as children of the hand/foot links, so every env gets
        # them and they move with the limbs (no render lag)
        rel_pivots = self._attach_thruster_visuals() if (self.cfg.debug_vis and self.cfg.thruster_vis == "attached") else []
        self.scene.clone_environments(copy_from_source=False)
        self._live_flames = (
            visuals.LiveFlames(self.cfg.robot.prim_path, rel_pivots, self.scene.cfg.num_envs)
            if rel_pivots and self.cfg.live_flames else None
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

    def _attach_thruster_visuals(self) -> list[str]:
        """A flame under each thruster link, starting at the nozzle and pointing along the exhaust (no casing: the
        jets come straight out of the soles and past the fingertips). Returns the flame pivots' paths relative to the
        robot root, or [] if the links can't be edited (then there are no thruster visuals)."""
        roots = visuals.robot_roots(self.cfg.robot.prim_path)
        if not roots:
            print(f"[R1IronMan] no robot prim at {self.cfg.robot.prim_path}: no thruster visuals")
            return []
        thr = self.cfg.attached_flame_throttle
        rel = []
        for root in roots:  # normally just env 0 (cloned afterwards)
            links = visuals.find_links(root, self.cfg.thruster_bodies)
            if links is None:
                print("[R1IronMan] can't attach thruster visuals: none drawn")
                return []
            rel = []
            for i, body in enumerate(self.cfg.thruster_bodies):
                pivot = visuals.add_flame(
                    str(links[body].GetPath()), visuals.FLAME_PIVOT, self.cfg.thruster_nozzles[i], self.cfg.thruster_exhaust[i],
                    radius=FLAME_RADIUS, length=FLAME_LENGTH[i] * thr, width=0.7 + 0.5 * thr,
                )
                rel.append(pivot[len(root) + 1:])
        print(f"[R1IronMan] flames attached to {list(self.cfg.thruster_bodies)} ({len(roots)} robot prim(s) before cloning)")
        return rel

    # ------------------------------------------------------------------ actions
    def _pre_physics_step(self, actions: torch.Tensor):
        self._prev_actions[:] = self._actions
        self._raw_actions = actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._throttle_cmd = (self._actions[:, 0:4] + 1.0) / 2.0

        targets = self._robot.data.default_joint_pos.torch.clone()
        targets[:, self._ctrl_joint_ids] += self._joint_scales * self._actions[:, 4:]
        self._robot.set_joint_position_target_index(target=targets)

        resample = (self.episode_length_buf % self._goal_resample_steps == 0) & (self.episode_length_buf > 0)
        ids = resample.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            self._sample_goals(ids)
            self._prev_dist[ids] = torch.linalg.norm(self._goal_pos_w[ids] - self._robot.data.root_pos_w.torch[ids], dim=1)

    def _thruster_frames(self):
        """World nozzle positions, thrust (push) directions and link quats, each (n, nt, ...)."""
        d = self._robot.data
        pos = d.body_link_pos_w.torch[:, self._thruster_body_ids]  # (n, nt, 3)
        quat = d.body_link_quat_w.torch[:, self._thruster_body_ids]  # (n, nt, 4)
        n, nt = pos.shape[:2]
        q = quat.reshape(-1, 4)
        nozzle = pos + quat_apply(q, self._nozzles_b.repeat(n, 1)).view(n, nt, 3)
        push = -quat_apply(q, self._exhaust_b.repeat(n, 1)).view(n, nt, 3)
        return nozzle, push, quat

    def _apply_action(self):
        self._throttle += (self._throttle_cmd - self._throttle) * min(1.0, self.physics_dt / self.cfg.throttle_time_constant)
        nozzle, push, _ = self._thruster_frames()
        force = push * (self._throttle * self._max_thrust).unsqueeze(-1)
        self._robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force,
            positions=nozzle,
            body_ids=self._thruster_body_ids,
            is_global=True,
        )

    # ------------------------------------------------------------------ MDP
    def _get_observations(self) -> dict:
        d = self._robot.data
        goal_b, _ = subtract_frame_transforms(d.root_pos_w.torch, d.root_quat_w.torch, self._goal_pos_w)
        height = d.root_pos_w.torch[:, 2:3] - self._terrain.env_origins[:, 2:3]
        _, push, _ = self._thruster_frames()
        nt = push.shape[1]
        push_b = quat_apply_inverse(d.root_quat_w.torch.repeat_interleave(nt, 0), push.reshape(-1, 3)).view(self.num_envs, -1)
        obs = torch.cat(
            [
                d.root_lin_vel_b.torch,
                d.root_ang_vel_b.torch,
                d.projected_gravity_b.torch,
                goal_b,
                height,
                self._throttle * 2.0 - 1.0,
                push_b,
                d.joint_pos.torch - d.default_joint_pos.torch,
                d.joint_vel.torch * 0.1,
                self._actions,
            ],
            dim=-1,
        )
        if self.cfg.heading_command:
            err = self._yaw_error()
            obs = torch.cat([obs, torch.sin(err).unsqueeze(-1), torch.cos(err).unsqueeze(-1)], dim=-1)
        return {"policy": obs}

    def _yaw(self) -> torch.Tensor:
        """Heading of the pelvis' forward axis in the world xy-plane."""
        q = self._robot.data.root_quat_w.torch
        fwd = quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(q.shape[0], 3))
        return torch.atan2(fwd[:, 1], fwd[:, 0])

    def _desired_yaw_rate(self) -> torch.Tensor:
        m = self.cfg.yaw_rate_tracking_max
        return (self.cfg.yaw_rate_tracking_gain * self._yaw_error()).clamp(-m, m)

    def _yaw_error(self) -> torch.Tensor:
        return torch.atan2(torch.sin(self._target_yaw - self._yaw()), torch.cos(self._target_yaw - self._yaw()))

    def _get_rewards(self) -> torch.Tensor:
        d = self._robot.data
        dist = torch.linalg.norm(self._goal_pos_w - d.root_pos_w.torch, dim=1)
        tilt = 1.0 + d.projected_gravity_b.torch[:, 2]
        jp = d.joint_pos.torch[:, self._ctrl_joint_ids] - d.default_joint_pos.torch[:, self._ctrl_joint_ids]
        joint_dev = torch.sum(torch.square(jp / self._joint_scales), dim=1)  # in units of each joint's action range
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
            "action_rate": torch.sum(torch.square(self._actions - self._prev_actions), dim=1)
            * self.cfg.action_rate_reward_scale
            * dt,
            "throttle": self._throttle.mean(dim=1) * self.cfg.throttle_reward_scale * dt,
            "joint_dev": joint_dev * self.cfg.joint_dev_reward_scale * dt / NUM_CONTROL_JOINTS,
            "joint_vel": torch.sum(torch.square(d.joint_vel.torch), dim=1) * self.cfg.joint_vel_reward_scale * dt,
            "heading": ((1.0 + torch.cos(self._yaw_error())) / 2.0) ** self.cfg.heading_sharpness
            * self.cfg.heading_reward_scale
            * dt,
            "yaw_rate": torch.square(d.root_ang_vel_w.torch[:, 2] - self._desired_yaw_rate()) * self.cfg.yaw_rate_reward_scale * dt,
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
        return fell | too_high | too_far | tilted, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)

        final_dist = torch.linalg.norm(self._goal_pos_w[env_ids] - self._robot.data.root_pos_w.torch[env_ids], dim=1).mean()
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
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        n = len(env_ids)
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        root_pose = self._robot.data.default_root_pose.torch[env_ids].clone()
        root_vel = torch.zeros(n, 6, device=self.device)
        root_pose[:, :3] += self._terrain.env_origins[env_ids]
        air = torch.rand(n, device=self.device) < self.cfg.air_start_prob
        zeros = torch.zeros(n, device=self.device)
        yaw = (torch.rand(n, device=self.device) * 2 - 1) * math.pi
        roll = torch.where(air, (torch.rand(n, device=self.device) * 2 - 1) * 0.3, zeros)
        pitch = torch.where(air, (torch.rand(n, device=self.device) * 2 - 1) * 0.3, zeros)
        root_pose[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
        n_air = int(air.sum())
        root_pose[air, 2] += torch.empty(n_air, device=self.device).uniform_(0.5, 2.5)
        root_vel[air, :3] = torch.empty(n_air, 3, device=self.device).uniform_(-1.0, 1.0)
        root_vel[air, 3:] = torch.empty(n_air, 3, device=self.device).uniform_(-0.5, 0.5)
        start = torch.where(air, self._hover_throttle, 0.0).unsqueeze(-1).expand(-1, self._throttle.shape[1])
        self._throttle[env_ids] = start
        self._throttle_cmd[env_ids] = start

        self._robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self._robot.write_joint_position_to_sim_index(position=self._robot.data.default_joint_pos.torch[env_ids], env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=self._robot.data.default_joint_vel.torch[env_ids], env_ids=env_ids)
        n = len(env_ids)
        r = self.cfg.goal_xy_range
        goals = torch.empty(n, 3, device=self.device)
        goals[:, :2] = torch.empty(n, 2, device=self.device).uniform_(-r, r)
        goals[:, 2] = torch.empty(n, device=self.device).uniform_(*self.cfg.goal_z_range)
        self.set_goals(env_ids, goals + self._terrain.env_origins[env_ids], pos_w=root_pose[:, :3])
        self._prev_dist[env_ids] = torch.linalg.norm(self._goal_pos_w[env_ids] - root_pose[:, :3], dim=1)

    def _sample_goals(self, env_ids: torch.Tensor):
        n = len(env_ids)
        r = self.cfg.goal_xy_range
        goals = torch.empty(n, 3, device=self.device)
        goals[:, :2] = torch.empty(n, 2, device=self.device).uniform_(-r, r)
        goals[:, 2] = torch.empty(n, device=self.device).uniform_(*self.cfg.goal_z_range)
        self.set_goals(env_ids, goals + self._terrain.env_origins[env_ids])

    def set_goals(self, env_ids: torch.Tensor, goals_w: torch.Tensor, pos_w: torch.Tensor | None = None):
        """Set new goals (world frame); the target heading points from pos_w (default: the robot) to the goal."""
        self._goal_pos_w[env_ids] = goals_w
        pos = self._robot.data.root_pos_w.torch[env_ids] if pos_w is None else pos_w
        delta = goals_w[:, :2] - pos[:, :2]
        far = delta.norm(dim=1) > self.cfg.heading_min_dist
        rand_yaw = (torch.rand(len(env_ids), device=self.device) * 2 - 1) * math.pi
        self._target_yaw[env_ids] = torch.where(far, torch.atan2(delta[:, 1], delta[:, 0]), rand_yaw)

    # ------------------------------------------------------------------ visuals
    def _set_debug_vis_impl(self, debug_vis: bool):
        # only the goal is a marker; the flames are attached to the robot (see _attach_thruster_visuals)
        if debug_vis and not hasattr(self, "_goal_vis"):
            self._goal_vis = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/R1IronMan/goal",
                    markers={
                        "goal": sim_utils.SphereCfg(
                            radius=0.12, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2))
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
            lengths = thr * torch.tensor(FLAME_LENGTH, device=self.device)
            self._live_flames.update(lengths.cpu(), (0.7 + 0.5 * thr).cpu())
