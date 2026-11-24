"""Config for the R1 "Iron Man" flight task.

Four thrusters, fixed to the robot's limbs, no gimbals:
  * one under each foot (``*_ankle_roll_link``), exhaust down out of the sole, thrust pushes along the foot's +z
  * one in each hand (``*_wrist_roll_link``), exhaust out past the fingertips, thrust pushes back up the forearm
The policy controls 4 throttles + 24 joint position offsets (legs, waist, arms) around an Iron Man flight pose:
legs straight, arms hanging down a little away from the body. It steers the way Iron Man does, by pointing its
feet (ankles act as gimbals) and swinging its arms. Thrusters are forces at the nozzle points of those links,
so their reaction loads go through the joints. Their casings and flames are visual markers only.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from r1_factory.robots.r1_hcrl import R1_HCRL  # isort: skip

R1_NUM_JOINTS = 26
# order is the order of the joint-offset actions; scale [rad] per unit action is given alongside
CONTROL_JOINTS = {
    ".*_hip_pitch_joint": 0.4,
    ".*_hip_roll_joint": 0.3,
    ".*_hip_yaw_joint": 0.3,
    ".*_knee_joint": 0.4,
    ".*_ankle_pitch_joint": 0.45,
    ".*_ankle_roll_joint": 0.25,
    "waist_.*": 0.2,
    ".*_shoulder_pitch_joint": 0.6,
    ".*_shoulder_roll_joint": 0.5,
    ".*_shoulder_yaw_joint": 0.4,
    ".*_elbow_joint": 0.5,
    ".*_wrist_roll_joint": 0.5,
}
NUM_CONTROL_JOINTS = 24

# Flight pose: straight legs, arms down and slightly out, forearms pointing down so the hand jets push up.
IRONMAN_POSE = {
    ".*_hip_pitch_joint": 0.0,
    ".*_knee_joint": 0.05,
    ".*_ankle_pitch_joint": -0.03,
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    ".*_shoulder_pitch_joint": 0.15,
    ".*_elbow_joint": 1.35,
}

_acts = R1_HCRL.actuators
R1_IRONMAN = R1_HCRL.replace(
    init_state=R1_HCRL.init_state.replace(pos=(0.0, 0.0, 0.79), joint_pos=IRONMAN_POSE),
    actuators={
        **_acts,
        # stiffer arms/ankles (a "suit"): the hand jets push ~70 N on a 0.45 m lever, the stock gains of 40 / 20
        # would let the limbs fold. Effort limits stay as in the robot config.
        "feet": _acts["feet"].replace(stiffness=80.0, damping=3.0),
        "shoulder": _acts["shoulder"].replace(stiffness=120.0, damping=6.0),
        "arms": _acts["arms"].replace(stiffness=100.0, damping=5.0),
    },
)


@configclass
class R1IronManEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 12.0
    decimation = 4  # 200 Hz physics, 50 Hz policy
    action_space = 4 + NUM_CONTROL_JOINTS
    # lin vel 3, ang vel 3, gravity 3, goal (body frame) 3, height 1, throttles 4, thrust dirs (body frame) 12,
    # joint pos 26, joint vel 26, last action 28
    observation_space = 29 + 2 * R1_NUM_JOINTS + 4 + NUM_CONTROL_JOINTS
    state_space = 0
    debug_vis = True
    # thruster flames (visual only): "attached" = cones parented to the hand/foot links, so they render with the robot
    # and never lag; "none" = no flames. Flames keep attached_flame_throttle's length unless every robot is in USD
    # (scene.clone_in_fabric=False, as scripts/showcase.py sets) and live_flames is on: then they follow the throttle.
    thruster_vis = "attached"
    attached_flame_throttle = 0.5  # hover is ~0.48
    # off in training: with 4096 robots in USD (newer Isaac Lab keeps them there) the per-step USD writes stall
    # training. scripts/showcase.py turns it on for its small grid.
    live_flames = False
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=12.0, replicate_physics=True, clone_in_fabric=True
    )

    robot: ArticulationCfg = R1_IRONMAN.replace(prim_path="/World/envs/env_.*/Robot")

    # ---- thrusters: (body, nozzle position in the body frame [m], exhaust direction in the body frame) --------
    # order: left foot, right foot, left hand, right hand
    thruster_bodies = ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_roll_link", "right_wrist_roll_link")
    thruster_nozzles = ((0.02, 0.0, -0.055), (0.02, 0.0, -0.055), (0.145, 0.0, 0.0), (0.145, 0.0, 0.0))
    thruster_exhaust = ((0.0, 0.0, -1.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    # max thrust of each thruster as a fraction of the robot's weight (total 2.1)
    thruster_max_twr = (0.75, 0.75, 0.30, 0.30)
    throttle_time_constant = 0.06  # [s]

    # ---- goals -----------------------------------------------------------------------------
    goal_xy_range = 3.0
    goal_z_range = (1.0, 4.0)
    goal_resample_time_s = 6.0
    air_start_prob = 0.5

    # ---- termination -----------------------------------------------------------------------
    min_height = 0.35
    max_height = 8.0
    max_tilt_cos = 0.25
    max_xy_dist = 6.0  # [m] from the env origin (was env_spacing / 2; kept separate so showcase can pack envs tighter)

    # ---- reward scales (multiplied by step_dt) ----------------------------------------------
    pos_coarse_reward_scale = 10.0
    pos_fine_reward_scale = 6.0
    progress_reward_scale = 3.0
    upright_reward_scale = -3.0
    lin_vel_reward_scale = -0.02
    ang_vel_reward_scale = -0.08
    action_rate_reward_scale = -0.01
    throttle_reward_scale = -0.05
    joint_dev_reward_scale = -0.1  # keep the Iron Man pose, loosely
    joint_vel_reward_scale = -0.0005
    action_bound_reward_scale = -1.0
    termination_penalty = -5.0  # one-off

    # ---- hold at the goal (stage 4) ------------------------------------------------------------
    # Penalties weighted by exp(-(dist / hold_radius)^2): full at the goal, ~0.37 at hold_radius, ~0 beyond 2x.
    # Limbs are Iron Man's steering, so motion is only made expensive once it has arrived. Off (0) before stage 4.
    hold_radius = 0.5  # [m]
    hold_ang_vel_reward_scale = 0.0  # * |body angular velocity|^2
    hold_joint_vel_reward_scale = 0.0  # * sum of controlled-joint speeds^2
    hold_action_rate_reward_scale = 0.0  # * |action - previous action|^2
    hold_pose_reward_scale = 0.0  # * mean squared distance from the hover pose, in units of each joint's action range
    # hover pose (offsets from the default pose, per controlled joint) recorded from a trained policy hovering at its
    # goals by scripts/hover_pose.py; "" = the default pose. Path relative to this folder.
    hover_pose_file = ""
    # low-pass filter on the joint position targets (first order, time constant tau): high-frequency limb jitter
    # can't reach the joints. 0 = off. 0.05 s = 29% new command per 20 ms policy step.
    joint_target_filter_tau = 0.0  # [s]
    # per-joint weights of the hold_pose term ({regex: weight}, unmatched joints 1.0; None = all 1.0)
    hold_pose_joint_weights = None
    # upright bonus at the goal, on the torso: hold * exp(-(torso tilt / width)^2). 0 = off (stages 1-4)
    hold_upright_reward_scale = 0.0
    hold_upright_width_deg = 3.0
    torso_body = "waist_yaw_link"

    # ---- heading command (v1 tasks): face the direction of the current goal -------------------
    heading_command = False  # adds sin/cos of the yaw error to the observations
    heading_reward_scale = 0.0  # * (1 + cos(yaw error)) / 2
    yaw_rate_reward_scale = 0.0  # * (world yaw rate - desired yaw rate)^2
    # desired yaw rate = clamp(gain * yaw error, +-max): turn toward the target heading at a calm, damped rate.
    # gain 0 = desired rate 0, i.e. a plain yaw-rate penalty (which fights turning toward the target)
    yaw_rate_tracking_gain = 0.0
    yaw_rate_tracking_max = 1.5  # [rad/s]
    heading_sharpness = 1.0  # reward = ((1 + cos(err)) / 2) ** sharpness; higher = more weight on small errors
    heading_min_dist = 1.0  # [m] goals closer than this horizontally get a random heading instead

    def __post_init__(self):
        self.observation_space = 29 + 2 * R1_NUM_JOINTS + self.action_space + (2 if self.heading_command else 0)


@configclass
class R1IronManHeadingEnvCfg(R1IronManEnvCfg):
    """Iron Man flies facing where it goes, instead of drifting/spinning in yaw."""

    heading_command = True
    heading_reward_scale = 2.0
    yaw_rate_reward_scale = -0.1


@configclass
class R1IronManHeadingSmoothEnvCfg(R1IronManHeadingEnvCfg):
    """Stage 2 for the heading task (--resume a stage-1 checkpoint)."""

    upright_reward_scale = -5.0
    ang_vel_reward_scale = -0.15
    action_rate_reward_scale = -0.04
    joint_dev_reward_scale = -0.3
    joint_vel_reward_scale = -0.002


@configclass
class R1IronManHoldEnvCfg(R1IronManHeadingSmoothEnvCfg):
    """Stage 4 (--resume the stage-3 final): settle at the goal instead of steering with constant limb motion.

    Stage 3's heading settings are built in (they were --env overrides). At the goal the stage-3 policy spends only
    ~2% of its reward on limb-motion penalties (joint speed RMS ~1.7 rad/s), so it never looks for a still pose.
    At the stage-3 level of fidgeting the hold terms cost ~4/s against ~16/s of position reward at the goal.
    """

    yaw_rate_reward_scale = -0.5
    yaw_rate_tracking_gain = 2.0
    heading_reward_scale = 3.0
    heading_sharpness = 4.0

    hold_ang_vel_reward_scale = -1.0
    hold_joint_vel_reward_scale = -0.03
    hold_action_rate_reward_scale = -0.5
    hold_pose_reward_scale = -3.0
    hover_pose_file = "hover_pose.json"
    joint_target_filter_tau = 0.05


@configclass
class R1IronManUprightEnvCfg(R1IronManHoldEnvCfg):
    """Stage 5 (--resume the stage-4 final): hold the natural upright pose at the goal, not its own learned one.

    Stage 4 pulled the limbs toward the pose stage 3 happened to hover in (hover_pose.json), which is lopsided
    (one arm flung out and back, hips uneven), so it froze a crooked stance about 5 deg off vertical. The target is
    now a designed pose (hover_pose_upright.json, from scripts/pose_check.py): legs straight and together, arms
    hanging straight down ~8 deg out, hand jets within 4 deg of vertical; checked to hover with the torso upright
    (feet + hand throttles balance it with <0.5 deg of lean). Ankles and wrists are weighted lightly: they are its
    gimbals for fine balance.
    """

    hover_pose_file = "hover_pose_upright.json"
    hold_pose_reward_scale = -6.0
    hold_pose_joint_weights = {".*_ankle_.*": 0.2, ".*_wrist_.*": 0.2}
    hold_upright_reward_scale = 4.0  # +4/s with the torso exactly vertical at the goal, +1.5/s at 3 deg, ~0 past 6
    hold_upright_width_deg = 3.0


@configclass
class R1IronManSmoothEnvCfg(R1IronManEnvCfg):
    """Stage 2 (--resume a stage-1 checkpoint): steadier, more upright, less flailing."""

    upright_reward_scale = -5.0
    ang_vel_reward_scale = -0.15
    action_rate_reward_scale = -0.04
    joint_dev_reward_scale = -0.3
    joint_vel_reward_scale = -0.002
