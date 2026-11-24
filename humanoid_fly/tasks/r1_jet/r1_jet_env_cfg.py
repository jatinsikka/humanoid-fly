"""Config for the R1 jet-flight task.

The R1 (29 kg) gets two jet boosters bolted to the back of its torso (``waist_yaw_link``).
Each booster has a throttle and a 2-axis gimbal (pitch about torso y, roll about torso x),
so the policy controls 6 numbers: [throttle_L, throttle_R, pitch_L, pitch_R, roll_L, roll_R].
The boosters are simulated as forces/torques on the torso link; their meshes are visual only.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from r1_factory.robots.r1_hcrl import R1_HCRL  # isort: skip

# R1 has 26 actuated joints: 12 legs, 2 waist, 10 arms, 2 head.
R1_NUM_JOINTS = 26
# Joints the whole-body variant may move (everything except the head): 24.
WHOLE_BODY_JOINTS = [".*_hip_.*", ".*_knee_joint", ".*_ankle_.*_joint", "waist_.*", ".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_roll_joint"]
WHOLE_BODY_NUM_JOINTS = 24


@configclass
class R1JetEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 12.0
    decimation = 4  # 200 Hz physics, 50 Hz policy
    action_space = 6
    observation_space = 0  # set in __post_init__
    state_space = 0
    debug_vis = True
    # jetpack visuals (visual only): "attached" = two booster casings + a pack plate bolted to the torso link, so they
    # move rigidly with the back and never lag, with a flame out of the bottom of each casing; "none" = nothing.
    # Flames keep attached_flame_throttle's length and point along the neutral axis, unless every robot is in USD
    # (scene.clone_in_fabric=False, as scripts/showcase.py sets) and live_flames is on: then they follow the
    # throttle and the gimbal.
    thruster_vis = "attached"
    attached_flame_throttle = 0.5  # hover = 1 / thrust_to_weight = 0.5
    live_flames = True
    # neutral booster axis for the casings [deg about torso y]. The physics computes it at startup from the CoM
    # (_compute_cant); the env prints a warning if this differs by more than 1 deg, with the value to put here.
    booster_vis_cant_deg = -13.5
    booster_vis_back_offset = 0.04  # [m] casings sit this much further back than the nozzle point (clear the torso)
    # training-clip camera (visuals.grid_camera): env 0 in front, the rest of the grid behind it [m]
    camera_back = 7.0
    camera_height = 3.0
    camera_ahead = 6.0
    camera_look_height = 1.8

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )
    # 12 m apart so flying robots from neighbouring envs never meet
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=12.0, replicate_physics=True, clone_in_fabric=True
    )

    robot: ArticulationCfg = R1_HCRL.replace(prim_path="/World/envs/env_.*/Robot")

    # ---- boosters --------------------------------------------------------------------------
    booster_body = "waist_yaw_link"
    # nozzle positions in the torso link frame [m]: on the back at shoulder-blade height, left/right of the spine.
    # v1 had them at (-0.14, +-0.12, 0.10): only 0.22 m above the whole-body CoM (which sits at z=-0.125, near the
    # hips), so pointing the thrust through the CoM needed a 38 deg gimbal (limit 25). No static hover existed and
    # every policy learned to spin at ~6 rad/s to average the torque out.
    # v2 (runs 10-11, checkpoints/jet/) had them 0.24 m apart (y = +-0.12): it flew, but rolled ~1.5 rad/s, since roll
    # comes mostly from differential throttle and that lever was short. v3 (2026-11-23): 0.40 m apart, ~1.7x the roll
    # torque. Old jet checkpoints were trained with +-0.12; to replay them faithfully set booster_mounts back.
    booster_mounts = ((-0.10, 0.20, 0.30), (-0.10, -0.20, 0.30))
    # cant the boosters' neutral axis so zero gimbal thrusts straight through the CoM (as real jetpacks do); the
    # robot then hovers leaning forward by the cant angle and keeps the full gimbal range for control
    cant_through_com = True
    thrust_to_weight = 2.0  # both boosters together at full throttle
    max_gimbal = math.radians(25.0)
    throttle_time_constant = 0.08  # [s] jet spool-up lag
    gimbal_time_constant = 0.04  # [s] gimbal servo lag

    # ---- whole-body control (off for the base task) ----------------------------------------
    control_joints = False
    joint_action_scale = 0.25  # [rad] offset from the default pose per unit action

    # ---- goals -----------------------------------------------------------------------------
    goal_xy_range = 3.0  # [m] +- around the env origin
    goal_z_range = (1.0, 4.0)  # [m] pelvis height
    goal_resample_time_s = 6.0
    air_start_prob = 0.5  # fraction of resets that start mid-air instead of standing

    # ---- termination -----------------------------------------------------------------------
    min_height = 0.35  # pelvis below this = fell over
    max_height = 8.0
    max_tilt_cos = 0.25  # terminate if the pelvis up-axis is tilted more than ~75 deg
    max_xy_dist = 6.0  # [m] from the env origin (was env_spacing / 2; kept separate so showcase can pack envs tighter)

    # ---- reward scales (multiplied by step_dt) ----------------------------------------------
    pos_coarse_reward_scale = 10.0
    pos_fine_reward_scale = 6.0
    progress_reward_scale = 3.0  # per m/s of speed toward the goal (clipped to +-2 m/s)
    upright_reward_scale = -3.0
    lin_vel_reward_scale = -0.02
    ang_vel_reward_scale = -0.08
    yaw_rate_reward_scale = 0.0  # extra penalty on spin about the body's vertical axis
    action_rate_reward_scale = -0.01
    throttle_reward_scale = -0.05
    joint_dev_reward_scale = -0.5
    # raw policy outputs are clamped to [-1, 1]; without this the means drift far past 1, the clamp kills
    # their gradient and the policy gets stuck in bang-bang control (run 5: 95% saturated, 6.5 rad/s spin)
    action_bound_reward_scale = -1.0
    termination_penalty = -5.0  # one-off, not scaled by dt

    def __post_init__(self):
        n_ctrl = WHOLE_BODY_NUM_JOINTS if self.control_joints else 0
        self.action_space = 6 + n_ctrl
        # lin vel 3, ang vel 3, gravity 3, goal (body frame) 3, height 1, jet state 6,
        # joint pos 26, joint vel 26, last action
        self.observation_space = 19 + 2 * R1_NUM_JOINTS + self.action_space


@configclass
class R1JetSmoothEnvCfg(R1JetEnvCfg):
    """Stage 2: fine-tune a flying stage-1 policy for a steady, upright, non-spinning hover."""

    upright_reward_scale = -5.0
    ang_vel_reward_scale = -0.15
    action_rate_reward_scale = -0.05
    # stage-1 policies spin about their vertical axis at ~6 rad/s (spin averages out the off-centre thrust);
    # at 6 rad/s this costs ~3.6/s, about a fifth of the position reward
    yaw_rate_reward_scale = -0.1


@configclass
class R1JetWholeBodyEnvCfg(R1JetEnvCfg):
    control_joints = True
