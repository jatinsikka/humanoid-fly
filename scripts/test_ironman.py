"""Smoke test for the Iron Man task: stand 1 s, then fire all 4 thrusters at a fixed throttle (no joint offsets).

    ~/IsaacLab/isaaclab.sh -p scripts/test_ironman.py --headless --enable_cameras --frames /tmp/frames
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", default="Humanoid-Fly-R1-IronMan-v0")
parser.add_argument("--throttle", type=float, default=0.55, help="all thrusters, [0, 1]; ~0.48 = hover")
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--frames", default=None, help="save a rendered png every 25 steps here")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import humanoid_fly  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
cfg.air_start_prob = 0.0
cfg.episode_length_s = 60.0
if args.frames:
    cfg.camera_eye_offset, cfg.camera_lookat_offset = (2.2, 2.2, 1.5), (0.0, 0.0, 1.0)
env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.frames else None)
uw = env.unwrapped
obs, _ = env.reset()
print("obs", obs["policy"].shape, "action dim", uw.cfg.action_space)
d = uw._robot.data
act = torch.zeros(args.num_envs, uw.cfg.action_space, device=uw.device)
act[:, 0:4] = -1.0
for t in range(50 + args.steps):
    if t == 50:
        act[:, 0:4] = 2 * args.throttle - 1
        nz, push, _ = uw._thruster_frames()
        print("nozzle heights", (nz[0, :, 2] - uw._terrain.env_origins[0, 2]).tolist())
        print("push dirs env0", push[0].tolist())
        print("body z of feet/hands", d.body_link_pos_w.torch[0, uw._thruster_body_ids, 2].tolist())
    obs, rew, term, trunc, _ = env.step(act)
    if t % 10 == 0:
        z = (d.root_pos_w.torch[:, 2] - uw._terrain.env_origins[:, 2]).mean().item()
        up = -d.projected_gravity_b.torch[:, 2].mean().item()
        print(f"t={t * uw.step_dt:5.2f}s  height={z:5.2f} m  up.z={up:+.2f}  angvel={d.root_ang_vel_b.torch.norm(dim=1).mean().item():.2f}  reward={rew.mean().item():+.3f}  dead={int(term.sum())}")
    if args.frames:
        img = env.render()  # every step, as the video recorder does
        if t % 25 == 0:
            import imageio

            os.makedirs(args.frames, exist_ok=True)
            imageio.imwrite(os.path.join(args.frames, f"f{t:04d}.png"), img)
    if not app.is_running():
        break
env.close()
app.close()
