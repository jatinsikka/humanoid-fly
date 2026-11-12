from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class R1IronManPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "r1_ironman"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.6),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class R1IronManSmoothPPORunnerCfg(R1IronManPPORunnerCfg):
    max_iterations = 1000
    experiment_name = "r1_ironman_smooth"

    def __post_init__(self):
        self.algorithm.entropy_coef = 0.0


@configclass
class R1IronManHeadingPPORunnerCfg(R1IronManPPORunnerCfg):
    max_iterations = 2000
    experiment_name = "r1_ironman_v1"

    def __post_init__(self):
        # v0 drifted to std 0.67 with the entropy bonus; flying is found within ~200 iterations anyway
        self.algorithm.entropy_coef = 0.002


@configclass
class R1IronManHeadingSmoothPPORunnerCfg(R1IronManPPORunnerCfg):
    max_iterations = 1000
    experiment_name = "r1_ironman_v1_smooth"

    def __post_init__(self):
        self.algorithm.entropy_coef = 0.0
