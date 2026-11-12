"""R1 with two gimballed jet boosters on its back, flying to 3D waypoints."""

import gymnasium as gym

from . import agents

gym.register(
    id="Humanoid-Fly-R1-Jet-v0",
    entry_point=f"{__name__}.r1_jet_env:R1JetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.r1_jet_env_cfg:R1JetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:R1JetPPORunnerCfg",
    },
)

# Stage 2: fine-tune a flying policy for a smooth, upright hover (train with --resume <stage-1 ckpt>).
gym.register(
    id="Humanoid-Fly-R1-Jet-Smooth-v0",
    entry_point=f"{__name__}.r1_jet_env:R1JetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.r1_jet_env_cfg:R1JetSmoothEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:R1JetSmoothPPORunnerCfg",
    },
)

# Same task, but the policy also drives the arms, legs and waist (on top of the jets).
gym.register(
    id="Humanoid-Fly-R1-Jet-WholeBody-v0",
    entry_point=f"{__name__}.r1_jet_env:R1JetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.r1_jet_env_cfg:R1JetWholeBodyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:R1JetWholeBodyPPORunnerCfg",
    },
)
