"""Watch a trained policy fly (opens the Isaac Sim GUI unless --headless).

    ~/IsaacLab/isaaclab.sh -p scripts/play.py                  # latest run, latest checkpoint
    ~/IsaacLab/isaaclab.sh -p scripts/play.py --checkpoint logs/r1_jet/<run>/model_1000.pt
"""

import argparse
import glob
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Humanoid-Fly-R1-Jet-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--steps", type=int, default=100000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import humanoid_fly  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env_cfg.scene.env_spacing = 6.0  # closer together so several fit on screen
env_cfg.goal_xy_range = 2.0
agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
agent_cfg.device = env_cfg.sim.device

ckpt = args.checkpoint
if ckpt is None:
    runs = sorted(glob.glob(os.path.join(PROJECT_DIR, "logs", agent_cfg.experiment_name, "*")))
    models = sorted(glob.glob(os.path.join(runs[-1], "model_*.pt")), key=lambda p: int(p.split("_")[-1][:-3]))
    ckpt = models[-1]
print(f"[INFO] loading {ckpt}")

env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(ckpt)
policy = runner.get_inference_policy(device=env.unwrapped.device)
export_dir = os.path.join(os.path.dirname(ckpt), "exported")
runner.export_policy_to_jit(path=export_dir, filename="policy.pt")

obs = env.get_observations()
u = env.unwrapped
falls, dist_sum, n = 0, 0.0, 0
for t in range(args.steps):
    if not app.is_running():
        break
    with torch.inference_mode():
        act = policy(obs)
        obs, _, dones, extras = env.step(act)
    falls += int(u.reset_terminated.sum())
    dist_sum += torch.linalg.norm(u._goal_pos_w - u._robot.data.root_pos_w.torch, dim=1).mean().item()
    n += 1
    if (t + 1) % 250 == 0:
        sat = (act.abs() > 0.98).float().mean().item()
        print(f"[EVAL] step {t + 1}: mean dist to goal {dist_sum / n:.3f} m, falls {falls}, "
              f"saturated actions {100 * sat:.0f}%, mean |ang vel| "
              f"{u._robot.data.root_ang_vel_b.torch.norm(dim=1).mean().item():.2f} rad/s "
              f"(|roll,pitch,yaw| {[round(x, 2) for x in u._robot.data.root_ang_vel_b.torch.abs().mean(0).tolist()]})")
        dist_sum, n = 0.0, 0
env.close()
app.close()
