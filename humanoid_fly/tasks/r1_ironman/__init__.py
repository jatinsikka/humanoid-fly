"""R1 with Iron Man-style thrusters in its feet and hands, flying to 3D waypoints."""

import gymnasium as gym

from . import agents

gym.register(
    id="Humanoid-Fly-R1-IronMan-v0",
    entry_point=f"{__name__}.r1_ironman_env:R1IronManEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.r1_ironman_env_cfg:R1IronManEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:R1IronManPPORunnerCfg",
    },
)

# Stage 2: fine-tune a flying policy for a steadier hover (train with --resume <stage-1 ckpt>).
gym.register(
    id="Humanoid-Fly-R1-IronMan-Smooth-v0",
    entry_point=f"{__name__}.r1_ironman_env:R1IronManEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.r1_ironman_env_cfg:R1IronManSmoothEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:R1IronManSmoothPPORunnerCfg",
    },
)

# v1: same robot, but also commanded to face the direction it flies (no yaw drift/spin).
gym.register(
    id="Humanoid-Fly-R1-IronMan-v1",
    entry_point=f"{__name__}.r1_ironman_env:R1IronManEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.r1_ironman_env_cfg:R1IronManHeadingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:R1IronManHeadingPPORunnerCfg",
    },
)

gym.register(
    id="Humanoid-Fly-R1-IronMan-Smooth-v1",
    entry_point=f"{__name__}.r1_ironman_env:R1IronManEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.r1_ironman_env_cfg:R1IronManHeadingSmoothEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:R1IronManHeadingSmoothPPORunnerCfg",
    },
)

# Stage 4: hold still at the goal (hold-at-goal penalties, hover-pose target, low-pass on joint targets);
# --resume the stage-3 final. Needs hover_pose.json next to the env (scripts/hover_pose.py).
gym.register(
    id="Humanoid-Fly-R1-IronMan-Hold-v1",
    entry_point=f"{__name__}.r1_ironman_env:R1IronManEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.r1_ironman_env_cfg:R1IronManHoldEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:R1IronManHoldPPORunnerCfg",
    },
)

# Stage 5: hold the natural upright pose at the goal (designed target pose, torso-upright bonus); --resume stage 4.
gym.register(
    id="Humanoid-Fly-R1-IronMan-Upright-v1",
    entry_point=f"{__name__}.r1_ironman_env:R1IronManEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.r1_ironman_env_cfg:R1IronManUprightEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:R1IronManUprightPPORunnerCfg",
    },
)
