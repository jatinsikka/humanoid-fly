"""Record the hover pose of a trained Iron Man policy: its average limb pose while holding at a goal.

Stage 4 (Humanoid-Fly-R1-IronMan-Hold-v1) pulls the limbs toward this pose near the goal, so the policy settles
into the pose it already hovers in instead of searching around it. Samples: robots within --max_dist of their goal,
in the second half of each goal, that haven't crashed (standing take-offs, random goals).

    ~/IsaacLab/isaaclab.sh -p scripts/hover_pose.py --headless --task Humanoid-Fly-R1-IronMan-Smooth-v1 \\
        --checkpoint <stage-3 final> --out humanoid_fly/tasks/r1_ironman/hover_pose.json

Writes {"offsets": {joint: rad from the default pose}, "std": {...}, "samples": n, ...}.
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--max_dist", type=float, default=0.1, help="[m] only count robots this close to their goal")
parser.add_argument("--seed", type=int, default=0)
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

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env_cfg.air_start_prob = 0.0
env_cfg.episode_length_s = args.seconds + 2.0
env_cfg.debug_vis = False
env_cfg.seed = args.seed
agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
agent_cfg.device = env_cfg.sim.device

env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
u = env.unwrapped
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=u.device)

ids = u._ctrl_joint_ids
names = [u._robot.joint_names[i] for i in ids]
goal_steps = int(u.cfg.goal_resample_time_s / u.step_dt)
steps = int(args.seconds / u.step_dt)
total = torch.zeros(len(ids), device=u.device, dtype=torch.float64)
total_sq = torch.zeros_like(total)
count = 0
with torch.inference_mode():
    torch.manual_seed(args.seed)
    env.reset()
    u.episode_length_buf[:] = 0
    obs = env.get_observations()
    crashed = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
    for t in range(steps):
        obs, _, _, _ = env.step(policy(obs))
        crashed |= u.reset_terminated
        if (t + 1) % goal_steps < goal_steps // 2:  # first half of each goal: still flying there
            continue
        d = u._robot.data
        near = (torch.linalg.norm(u._goal_pos_w - d.root_pos_w.torch, dim=1) < args.max_dist) & ~crashed
        if near.any():
            jp = (d.joint_pos.torch[near][:, ids] - d.default_joint_pos.torch[near][:, ids]).double()
            total += jp.sum(0)
            total_sq += (jp * jp).sum(0)
            count += int(near.sum())

assert count > 0, "no robot ever got within --max_dist of its goal"
mean = total / count
std = (total_sq / count - mean * mean).clamp(min=0).sqrt()
out = {
    "offsets": {n: round(float(m), 5) for n, m in zip(names, mean)},
    "std": {n: round(float(s), 5) for n, s in zip(names, std)},
    "samples": count,
    "task": args.task,
    "checkpoint": args.checkpoint,
    "max_dist_m": args.max_dist,
}
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(out, f, indent=2)
print(f"[HOVER_POSE] {count} samples -> {args.out}")
for n in names:
    print(f"  {n:28s} {out['offsets'][n]:+.3f} rad  (std {out['std'][n]:.3f})")
env.close()
app.close()
