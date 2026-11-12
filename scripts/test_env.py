"""Smoke test: build the env, fire both boosters at a fixed throttle, print what happens.

    ~/IsaacLab/isaaclab.sh -p scripts/test_env.py --headless --num_envs 16
    ~/IsaacLab/isaaclab.sh -p scripts/test_env.py --num_envs 4 --throttle 0.62   # watch it in the GUI
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", default="Humanoid-Fly-R1-Jet-v0")
parser.add_argument("--throttle", type=float, default=0.6, help="booster throttle in [0, 1]; 0.5 = hover")
parser.add_argument("--steps", type=int, default=150)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import humanoid_fly  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
cfg.air_start_prob = 0.0  # everyone starts standing
env = gym.make(args.task, cfg=cfg)
uw = env.unwrapped
obs, _ = env.reset()
print("obs", obs["policy"].shape, "action dim", uw.cfg.action_space)
act = torch.zeros(args.num_envs, uw.cfg.action_space, device=uw.device)
act[:, 0:2] = 2 * args.throttle - 1
for t in range(args.steps):
    obs, rew, term, trunc, _ = env.step(act)
    if t % 10 == 0:
        d = uw._robot.data
        z = (d.root_pos_w.torch[:, 2] - uw._terrain.env_origins[:, 2]).mean().item()
        up = -d.projected_gravity_b.torch[:, 2].mean().item()
        print(f"t={t * uw.step_dt:5.2f}s  height={z:5.2f} m  up·z={up:+.2f}  reward={rew.mean().item():+.3f}  dead={int(term.sum())}")
    if not app.is_running():
        break
env.close()
app.close()
