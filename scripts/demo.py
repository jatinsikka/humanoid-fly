"""Record a demo flight: take off from the ground and fly a fixed waypoint tour, with a chase camera.

Saves an mp4 next to the checkpoint and (with --wandb) uploads it to the W&B project humanoid_fly.

    ~/IsaacLab/isaaclab.sh -p scripts/demo.py --headless --task Humanoid-Fly-R1-IronMan-v0 --wandb
    ~/IsaacLab/isaaclab.sh -p scripts/demo.py --headless --checkpoint logs/r1_ironman/<run>/model_3000.pt
"""

import argparse
import glob
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Humanoid-Fly-R1-IronMan-v0")
parser.add_argument("--checkpoint", default=None, help="default: latest checkpoint of the task's newest run")
parser.add_argument("--out", default=None, help="mp4 path (default: <run>/demo_<ckpt>.mp4)")
parser.add_argument("--hold_s", type=float, default=4.0, help="seconds per waypoint")
parser.add_argument("--wandb", action="store_true", help="upload the video + metrics to W&B project humanoid_fly")
parser.add_argument("--no_video", action="store_true", help="metrics only")
parser.add_argument("--num_envs", type=int, default=1, help="fly the tour in this many envs at once (metrics over all, video of env 0)")
parser.add_argument("--thruster_vis", default=None, choices=["attached", "none"],
                    help="flames / jetpack drawn attached to the robot (default) or not at all")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = not args.no_video
app = AppLauncher(args).app

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import humanoid_fly  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# waypoints relative to the start position [m]: lift off, fly a loop around the field, come back
WAYPOINTS = [(0, 0, 2.0), (2.5, 0, 3.0), (2.5, 2.5, 2.0), (-2.0, 2.5, 3.5), (-2.5, -2.0, 1.5), (1.5, -2.5, 2.5), (0, 0, 1.5)]

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env_cfg.air_start_prob = 0.0
env_cfg.goal_resample_time_s = 1e6
env_cfg.episode_length_s = args.hold_s * len(WAYPOINTS) + 2.0
if args.thruster_vis is not None:
    if not hasattr(env_cfg, "thruster_vis"):
        raise SystemExit(f"--thruster_vis is not supported by {args.task}")
    env_cfg.thruster_vis = args.thruster_vis
agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
agent_cfg.device = env_cfg.sim.device
if not args.no_video:
    env_cfg.video_recorder.window_width, env_cfg.video_recorder.window_height = 1280, 720

ckpt = args.checkpoint
if ckpt is None:
    runs = sorted(glob.glob(os.path.join(PROJECT_DIR, "logs", agent_cfg.experiment_name, "*")))
    runs = [r for r in runs if glob.glob(os.path.join(r, "model_*.pt"))]
    ckpt = sorted(glob.glob(os.path.join(runs[-1], "model_*.pt")), key=lambda p: int(p.split("_")[-1][:-3]))[-1]
print(f"[INFO] loading {ckpt}")

env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg, render_mode=None if args.no_video else "rgb_array"))
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(ckpt)
policy = runner.get_inference_policy(device=env.unwrapped.device)
u = env.unwrapped

obs = env.get_observations()
u.episode_length_buf[:] = 0  # the env staggers episode lengths at startup; we want one full tour, no timeout reset
N = u.num_envs
origin = u._terrain.env_origins.clone()
start = u._robot.data.root_pos_w.torch.clone()
# (n_wp, N, 3): the same tour for every env, relative to its own start
wps = torch.stack([torch.stack([start[:, 0] + x, start[:, 1] + y, origin[:, 2] + z], dim=-1) for x, y, z in WAYPOINTS])
hold = int(args.hold_s / u.step_dt)
all_ids = torch.arange(N, device=u.device)
heading = getattr(u.cfg, "heading_command", False)


def set_goal(i):
    if hasattr(u, "set_goals"):  # also sets the heading target (face the next waypoint)
        u.set_goals(all_ids, wps[i])
    else:
        u._goal_pos_w[:] = wps[i]
    u._prev_dist[:] = torch.linalg.norm(wps[i] - u._robot.data.root_pos_w.torch, dim=1)


set_goal(0)
alive = torch.ones(N, dtype=torch.bool, device=u.device)
acc = {k: [] for k in ["dist", "hold_yaw", "hold_yaw_err", "tilt", "angvel", "yawrate", "jdev", "yaw_err"]}
frames, end_err = [], []
cam_target = start[0].cpu().numpy() + np.array([0, 0, 1.0])
if not args.no_video:
    from isaacsim.core.rendering_manager import ViewportManager


def add(key, v, holding=False):
    acc[key].append(v[alive].float().cpu())


for t in range(hold * len(WAYPOINTS)):
    if t % hold == 0:
        set_goal(t // hold)
    if t % hold == hold - 1:  # end of each hold: who got there
        end_err.append(torch.linalg.norm(wps[t // hold] - u._robot.data.root_pos_w.torch, dim=1)[alive].cpu())
    with torch.inference_mode():
        obs, _, dones, _ = env.step(policy(obs))
    crashed_now = u.reset_terminated & alive
    if bool(crashed_now[0]):
        print(f"[DEMO] env 0 crashed at t={t * u.step_dt:.2f}s")
    alive &= ~u.reset_terminated
    if not bool(alive.any()):
        break
    d = u._robot.data
    w = d.root_ang_vel_w.torch
    if t % hold > hold // 2:  # second half of each hold: how well does it hold the point
        add("dist", torch.linalg.norm(wps[t // hold] - d.root_pos_w.torch, dim=1))
        add("hold_yaw", w[:, 2].abs())
        if heading:
            add("hold_yaw_err", torch.rad2deg(u._yaw_error()).abs())
    add("angvel", w.norm(dim=1))
    add("yawrate", w[:, 2].abs())
    add("jdev", (d.joint_pos.torch - d.default_joint_pos.torch).abs().mean(dim=1))
    if heading:
        add("yaw_err", torch.rad2deg(u._yaw_error()).abs())
    add("tilt", torch.rad2deg(torch.acos((-d.projected_gravity_b.torch[:, 2]).clamp(-1, 1))))
    if not args.no_video and t % 2 == 0 and bool(alive[0]):  # 25 fps, follow env 0
        cam_target = 0.92 * cam_target + 0.08 * d.root_pos_w.torch[0].cpu().numpy()  # smooth chase
        ang = 0.6 + 0.12 * t * u.step_dt  # slow orbit
        eye = cam_target + np.array([4.5 * np.cos(ang), 4.5 * np.sin(ang), 1.0])
        ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=eye.tolist(), target=cam_target.tolist())
        frames.append(env.unwrapped.render())

cat = {k: torch.cat(v).numpy() if v else np.array([np.nan]) for k, v in acc.items()}
metrics = {
    "demo/num_envs": N,
    "demo/crash_rate": 1.0 - alive.float().mean().item(),
    "demo/hold_error_mean_m": float(np.mean(cat["dist"])),
    "demo/hold_error_median_m": float(np.median(cat["dist"])),
    "demo/hold_error_p95_m": float(np.percentile(cat["dist"], 95)),
    "demo/tilt_mean_deg": float(np.mean(cat["tilt"])),
    "demo/tilt_p99_deg": float(np.percentile(cat["tilt"], 99)),
    "demo/ang_vel_mean": float(np.mean(cat["angvel"])),
    "demo/yaw_rate_mean": float(np.mean(cat["yawrate"])),
    "demo/joint_dev_mean_rad": float(np.mean(cat["jdev"])),
    "demo/hold_yaw_rate_mean": float(np.mean(cat["hold_yaw"])),
    "demo/hold_heading_error_mean_deg": float(np.mean(cat["hold_yaw_err"])),
    "demo/heading_error_mean_deg": float(np.mean(cat["yaw_err"])),
}
for i, e in enumerate(end_err):
    metrics[f"demo/wp{i}_reached_frac"] = (e < 0.3).float().mean().item()
print("[DEMO]", ckpt, {k: round(v, 3) for k, v in metrics.items()})
if len(end_err) == len(WAYPOINTS) and alive.all():
    E = torch.stack(end_err)  # (n_wp, N)
    fails = (E >= 0.3).sum(0)
    print("[DEMO] envs by #waypoints missed:", torch.bincount(fails, minlength=len(WAYPOINTS) + 1).tolist())
    if bool((E >= 0.3).any()):
        print("[DEMO] missed end errors (quantiles 10/50/90):", torch.quantile(E[E >= 0.3], torch.tensor([0.1, 0.5, 0.9])).tolist())
    bad = int(torch.argmax(fails))
    print(f"[DEMO] worst env {bad}: end errors", [round(x, 2) for x in E[:, bad].tolist()])
print("[DEMO] reached (<0.3 m at end of hold) per waypoint:", [round((e < 0.3).float().mean().item(), 2) for e in end_err],
      "median end err:", [round(e.median().item(), 2) for e in end_err])

if frames:
    import imageio

    out = args.out or os.path.join(os.path.dirname(ckpt), f"demo_{os.path.basename(ckpt)[:-3]}.mp4")
    imageio.mimwrite(out, frames[3:], fps=25, quality=8)  # first frames can be blank during warmup
    print(f"[DEMO] wrote {out}")
    if args.wandb:
        import wandb

        run = wandb.init(project="humanoid_fly", job_type="demo", name=f"demo-{args.task}-{os.path.basename(os.path.dirname(ckpt))}-{os.path.basename(ckpt)[:-3]}",
                         config={"task": args.task, "checkpoint": ckpt, "waypoints": WAYPOINTS})
        wandb.log({**metrics, "demo/video": wandb.Video(out, format="mp4")})
        run.finish()
env.close()
app.close()
