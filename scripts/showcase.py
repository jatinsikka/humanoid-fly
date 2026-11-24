"""Before/after showcase video: a grid of robots taking off from the ground, one clip per checkpoint.

Same seed, starts and goals for every checkpoint, so the clips compare directly (e.g. iteration 0 vs the final policy,
or a learning-to-balance series 0, 10, 25, 45, final). Flames follow each robot's throttle (and the jet's gimbal)
because every robot is kept in USD here (scene.clone_in_fabric=False), which is fine for a few dozen robots.
Robots that crash are reset standing on the ground, as in training.

    ~/IsaacLab/isaaclab.sh -p scripts/showcase.py --headless --task Humanoid-Fly-R1-IronMan-Smooth-v1 \\
        --checkpoints logs/r1_ironman_v1/<run>/model_0.pt checkpoints/ironman/model_2599.pt \\
        --labels "Iteration 0" "Final (iteration 2599)" --out_dir media/showcase/ironman

Writes into --out_dir: clip_<i>.mp4 per checkpoint, before_after.mp4 (first vs last, side by side),
before_after.gif + clip_<i>.gif (small, for web pages), and metrics.json (crashes per checkpoint).
"""

import argparse
import ast
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoints", nargs="+", required=True, help="model_*.pt files, in the order to show")
parser.add_argument("--labels", nargs="*", default=None, help="one caption per checkpoint (default: file names)")
parser.add_argument("--out_dir", required=True)
parser.add_argument("--num_envs", type=int, default=36, help="robots in the grid")
parser.add_argument("--env_spacing", type=float, default=4.5, help="[m] between robots (training uses 12)")
parser.add_argument("--goal_xy", type=float, default=1.2, help="[m] goals within +-this of each robot's start")
parser.add_argument("--goal_z", type=float, nargs=2, default=(1.2, 2.8), help="[m] goal height range")
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--air_start_prob", type=float, default=0.0, help="0 = everyone takes off from the ground")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--camera", type=float, nargs=4, default=(6.0, 2.5, 6.0, 1.6), metavar=("BACK", "HEIGHT", "AHEAD", "LOOK_H"),
                    help="visuals.grid_camera: distance behind env 0, eye height, look-at distance past env 0, look-at height")
parser.add_argument("--gif_width", type=int, default=480)
parser.add_argument("--env", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override env cfg fields (Python literals), e.g. the old jet mount for old checkpoints")
parser.add_argument("--wandb", action="store_true", help="also upload the videos + metrics to W&B (project humanoid_fly)")
parser.add_argument("--no_live_flames", action="store_true", help="fixed flame length (as in training clips)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import humanoid_fly  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402

FPS = 25
labels = args.labels or [os.path.basename(c)[:-3] for c in args.checkpoints]
assert len(labels) == len(args.checkpoints), "need one label per checkpoint"
os.makedirs(args.out_dir, exist_ok=True)

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
for kv in args.env:
    k, v = kv.split("=", 1)
    assert hasattr(env_cfg, k), f"env cfg has no field {k}"
    try:
        v = ast.literal_eval(v)
    except (ValueError, SyntaxError):
        pass
    setattr(env_cfg, k, v)
    print(f"[SHOWCASE] env override {k} = {v!r}")
env_cfg.scene.env_spacing = args.env_spacing
env_cfg.scene.clone_in_fabric = False  # every robot in USD, so flames can follow each robot's throttle
env_cfg.live_flames = not args.no_live_flames
env_cfg.air_start_prob = args.air_start_prob
env_cfg.goal_xy_range = args.goal_xy
env_cfg.goal_z_range = tuple(args.goal_z)
env_cfg.episode_length_s = args.seconds + 2.0  # no timeouts during the clip
env_cfg.camera_back, env_cfg.camera_height, env_cfg.camera_ahead, env_cfg.camera_look_height = args.camera
env_cfg.video_recorder.window_width, env_cfg.video_recorder.window_height = args.width, args.height
env_cfg.seed = args.seed
agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
agent_cfg.device = env_cfg.sim.device

env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg, render_mode="rgb_array"))
u = env.unwrapped
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
steps = int(args.seconds / u.step_dt)
every = max(1, round(1.0 / (u.step_dt * FPS)))


# ------------------------------------------------------------------ captions + resizing (PIL if present)
def _font(size):
    try:
        from PIL import ImageFont

        try:
            return ImageFont.load_default(size=size)
        except TypeError:  # Pillow < 10.1
            return ImageFont.load_default()
    except ImportError:
        return None


def caption(frame, text):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return frame
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    size = max(18, frame.shape[0] // 18)
    f = _font(size)
    pad = size // 3
    box = d.textbbox((0, 0), text, font=f)
    d.rectangle((0, 0, box[2] + 3 * pad, box[3] + 3 * pad), fill=(0, 0, 0))
    d.text((pad * 1.5, pad), text, fill=(255, 255, 255), font=f)
    return np.asarray(img)


def resize(frame, width):
    if frame.shape[1] == width:
        return frame
    h = round(frame.shape[0] * width / frame.shape[1])
    try:
        from PIL import Image

        return np.asarray(Image.fromarray(frame).resize((width, h), Image.LANCZOS))
    except ImportError:
        k = max(1, frame.shape[1] // width)
        return frame[::k, ::k]


def write(path, frames, fps):
    if path.endswith(".gif"):
        try:  # Pillow: duration is always in ms (imageio's GIF duration unit differs between versions)
            from PIL import Image

            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=round(1000 / fps), loop=0)
        except ImportError:
            imageio.mimwrite(path, frames, duration=1000 / fps, loop=0)
    else:
        imageio.mimwrite(path, frames, fps=fps, quality=8)
    print(f"[SHOWCASE] wrote {path}")


# ------------------------------------------------------------------ one clip per checkpoint
clips, metrics = [], {}
for i, (ckpt, label) in enumerate(zip(args.checkpoints, labels)):
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=u.device)
    torch.manual_seed(args.seed)  # same starts and goals for every checkpoint
    with torch.inference_mode():  # env buffers are inference tensors after the first clip; reset them in that mode
        env.reset()
        u.episode_length_buf[:] = 0  # the env staggers episode lengths on a full reset; we want one clean clip
        obs = env.get_observations()
    for _ in range(3):  # renderer warm-up (first frames can be blank)
        u.render()
    frames, crashes, airborne = [], 0, []
    for t in range(steps):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        crashes += int(u.reset_terminated.sum())
        h = u._robot.data.root_pos_w.torch[:, 2] - u._terrain.env_origins[:, 2]
        airborne.append((h > 1.0).float().mean().item())
        if t % every == 0:
            frames.append(caption(u.render(), label))
    clips.append(frames)
    metrics[label] = {
        "checkpoint": ckpt,
        "robots": args.num_envs,
        "seconds": args.seconds,
        "crashes": crashes,
        "crashes_per_robot": crashes / args.num_envs,
        "fraction_above_1m_last_half": float(np.mean(airborne[len(airborne) // 2:])),
    }
    print(f"[SHOWCASE] {label}: {metrics[label]}")
    write(os.path.join(args.out_dir, f"clip_{i}.mp4"), frames, FPS)
    write(os.path.join(args.out_dir, f"clip_{i}.gif"), [resize(f, args.gif_width) for f in frames[::2]], FPS / 2)

if len(clips) >= 2:
    half = args.width // 2
    n = min(len(clips[0]), len(clips[-1]))
    sbs = [np.concatenate([resize(clips[0][k], half), resize(clips[-1][k], half)], axis=1) for k in range(n)]
    write(os.path.join(args.out_dir, "before_after.mp4"), sbs, FPS)
    write(os.path.join(args.out_dir, "before_after.gif"), [resize(f, 2 * args.gif_width) for f in sbs[::2]], FPS / 2)

with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
    json.dump({"task": args.task, "seed": args.seed, "air_start_prob": args.air_start_prob, "clips": metrics}, f, indent=2)
print(f"[SHOWCASE] metrics in {os.path.join(args.out_dir, 'metrics.json')}")
if args.wandb:
    import wandb

    run = wandb.init(project="humanoid_fly", job_type="showcase", name=f"showcase-{args.task}",
                     config={"task": args.task, "checkpoints": args.checkpoints, "labels": labels, "seed": args.seed})
    media = {f"showcase/{label}": wandb.Video(os.path.join(args.out_dir, f"clip_{i}.mp4"), format="mp4")
             for i, label in enumerate(labels)}
    if len(clips) >= 2:
        media["showcase/before_after"] = wandb.Video(os.path.join(args.out_dir, "before_after.mp4"), format="mp4")
    wandb.log({**media, **{f"showcase/{lbl}/{k}": v for lbl, m in metrics.items() for k, v in m.items()
                           if isinstance(v, (int, float))}})
    run.finish()
env.close()
app.close()
