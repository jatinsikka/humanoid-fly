"""Train an R1 jet-flight policy with RSL-RL.

Our own launcher because Isaac Lab 3.0's scripts/reinforcement_learning/rsl_rl/train.py crashes on this
machine (free(): invalid pointer at Kit startup, even for the stock quadcopter task). Starting the app
first, as below, works.

    ~/IsaacLab/isaaclab.sh -p scripts/train.py --headless          # logs + clips to W&B project humanoid_fly
    ~/IsaacLab/isaaclab.sh -p scripts/train.py --headless --task Humanoid-Fly-R1-Jet-WholeBody-v0
    ~/IsaacLab/isaaclab.sh -p scripts/train.py --headless --resume logs/r1_jet/<run>/model_500.pt
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Humanoid-Fly-R1-Jet-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", default=None, help="checkpoint .pt to continue from")
parser.add_argument("--reset_std", type=float, default=None, help="with --resume: reset the action noise std to this")
parser.add_argument("--no_video", action="store_true", help="skip recording clips (faster)")
parser.add_argument("--video_length", type=int, default=300, help="clip length in policy steps (50 per second)")
parser.add_argument("--video_interval", type=int, default=2400, help="policy steps between clips (24 per iteration)")
parser.add_argument("--env", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override env cfg fields, e.g. --env yaw_rate_reward_scale=-0.4 upright_reward_scale=-6")
parser.add_argument("--logger", default="wandb", choices=["wandb", "tensorboard"])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = not args.no_video
app = AppLauncher(args).app

import importlib.metadata as metadata  # noqa: E402
import os  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import humanoid_fly  # noqa: E402,F401
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
for kv in args.env:
    k, v = kv.split("=", 1)
    assert hasattr(env_cfg, k), f"env cfg has no field {k}"
    setattr(env_cfg, k, type(getattr(env_cfg, k))(v))
    print(f"[INFO] env override {k} = {getattr(env_cfg, k)}")
agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
if args.max_iterations is not None:
    agent_cfg.max_iterations = args.max_iterations
agent_cfg.seed = env_cfg.seed = args.seed
agent_cfg.device = env_cfg.sim.device
agent_cfg.logger = args.logger
agent_cfg.wandb_project = "humanoid_fly"
env_cfg.video_recorder.window_width, env_cfg.video_recorder.window_height = 960, 540  # camera aim: env _setup_scene

log_dir = os.path.join(PROJECT_DIR, "logs", agent_cfg.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
env_cfg.log_dir = log_dir
print(f"[INFO] logging to {log_dir}")

env = gym.make(args.task, cfg=env_cfg, render_mode=None if args.no_video else "rgb_array")
if not args.no_video:
    # rsl_rl uploads every .mp4 under log_dir to W&B (Media panel, key "video")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=os.path.join(log_dir, "videos", "train"),
        step_trigger=lambda step: step % args.video_interval == 0,
        video_length=args.video_length,
        disable_logger=True,
    )
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
if args.resume:
    runner.load(args.resume)
    if args.reset_std is not None:
        import torch

        n_reset = 0
        for module in [m for m in vars(runner.alg).values() if isinstance(m, torch.nn.Module)]:
            for name, p in module.named_parameters():
                if name.endswith("log_std_param"):
                    p.data.fill_(float(torch.log(torch.tensor(args.reset_std))))
                    n_reset += 1
                elif name.endswith("std_param"):
                    p.data.fill_(args.reset_std)
                    n_reset += 1
        print(f"[INFO] reset action std to {args.reset_std} ({n_reset} parameter(s))")
        assert n_reset > 0, "no std parameter found to reset"
dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
env.close()
app.close()
