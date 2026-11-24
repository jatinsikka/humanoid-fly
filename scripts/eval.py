"""Robustness evaluation: one checkpoint, several conditions, one table.

scripts/demo.py flies one fixed waypoint tour from a standing take-off (only the starting heading varies), which
says "it flies this tour" and little more. This runs N robots for one 12 s episode under each condition:

  takeoff        standing start, random goals (training ranges, a new goal every 6 s)
  midair         start in the air: random tilt (+-17 deg), velocity (+-1 m/s) and spin (+-0.5 rad/s)
  pushes         standing start, random goals, plus a random horizontal velocity kick every ~2 s
  thrust_low     all thrusters 10% weaker (roughly an 11% heavier robot)
  thrust_high    all thrusters 10% stronger
  obs_noise      Gaussian noise on every observation the policy sees

Training only ever saw the first two (no pushes, no model error, no sensor noise), so expect the others to be worse:
that is the point, it tells you what to train on next. The policy is deterministic (mean action).

    ~/IsaacLab/isaaclab.sh -p scripts/eval.py --headless --task Humanoid-Fly-R1-IronMan-Smooth-v1 \\
        --checkpoint checkpoints/ironman/model_2599.pt --out media/eval/ironman
    # replay the old (v2) jet policy with the mount it was trained with:
    ~/IsaacLab/isaaclab.sh -p scripts/eval.py --headless --task Humanoid-Fly-R1-Jet-Smooth-v0 \\
        --checkpoint checkpoints/jet/model_1499.pt --out media/eval/jet_v2 \\
        --env "booster_mounts=((-0.10, 0.12, 0.30), (-0.10, -0.12, 0.30))"

Writes <out>.json and <out>.md (the table), and with --wandb logs both to W&B (project humanoid_fly, job eval).
"""

import argparse
import ast
import json
import math
import os

from isaaclab.app import AppLauncher

CONDITIONS = ["takeoff", "midair", "pushes", "thrust_low", "thrust_high", "obs_noise"]

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True, help="output prefix, e.g. media/eval/ironman")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--seconds", type=float, default=12.0, help="one episode per robot per condition")
parser.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
parser.add_argument("--push_dv", type=float, nargs=2, default=(0.5, 1.5), help="[m/s] range of each velocity kick")
parser.add_argument("--push_every", type=float, default=2.0, help="[s] mean time between kicks (per robot, random)")
parser.add_argument("--thrust_delta", type=float, default=0.10)
parser.add_argument("--obs_noise", type=float, default=0.05, help="std of the Gaussian noise added to observations")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--env", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override env cfg fields (Python literals), e.g. \"booster_mounts=((-0.1, 0.12, 0.3), (-0.1, -0.12, 0.3))\"")
parser.add_argument("--wandb", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import humanoid_fly  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
for kv in args.env:
    k, v = kv.split("=", 1)
    assert hasattr(env_cfg, k), f"env cfg has no field {k}"
    try:
        val = ast.literal_eval(v)
    except (ValueError, SyntaxError):
        val = v
    setattr(env_cfg, k, val)
    print(f"[EVAL] env override {k} = {val!r}")
env_cfg.debug_vis = False  # no rendering needed
env_cfg.episode_length_s = args.seconds + 2.0  # no timeouts during the measurement
env_cfg.seed = args.seed
agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
agent_cfg.device = env_cfg.sim.device

env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
u = env.unwrapped
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=u.device)
N, dt, dev = u.num_envs, u.step_dt, u.device
steps = int(args.seconds / dt)
goal_steps = u._goal_resample_steps
settle = int(1.5 / dt)  # hover error = last 1.5 s before each goal switch
base_air_prob = u.cfg.air_start_prob
base_thrust = u._max_thrust.clone() if torch.is_tensor(u._max_thrust) else u._max_thrust


def root_vel(d):
    for name in ("root_com_vel_w", "root_vel_w", "root_link_vel_w"):
        if hasattr(type(d), name):
            return getattr(d, name).torch.clone()
    raise AttributeError("no root velocity on robot data")


def noisy(obs, std):
    if std <= 0:
        return obs
    if torch.is_tensor(obs):
        return obs + std * torch.randn_like(obs)
    obs = obs.clone() if hasattr(obs, "clone") else dict(obs)  # TensorDict / dict of observation groups
    obs["policy"] = obs["policy"] + std * torch.randn_like(obs["policy"])
    return obs


# the whole condition runs in inference mode: the env's buffers become inference tensors once the policy steps
# under inference_mode, and resetting them outside it raises "Inplace update to inference tensor"
@torch.inference_mode()
def run(cond):
    u.cfg.air_start_prob = 1.0 if cond == "midair" else 0.0
    scale = {"thrust_low": 1.0 - args.thrust_delta, "thrust_high": 1.0 + args.thrust_delta}.get(cond, 1.0)
    u._max_thrust = base_thrust * scale
    noise = args.obs_noise if cond == "obs_noise" else 0.0
    torch.manual_seed(args.seed)
    env.reset()
    u.episode_length_buf[:] = 0
    obs = env.get_observations()

    crashed = torch.zeros(N, dtype=torch.bool, device=dev)  # crashed at least once (later episodes don't count)
    n_crashes = 0
    errs, reached, tilts, angv, rolls, jspd, dact = [], [], [], [], [], [], []
    recovered_at = torch.full((N,), float("nan"), device=dev)
    calm_steps = torch.zeros(N, device=dev)
    next_push = torch.rand(N, device=dev) * 2 * args.push_every
    for t in range(steps):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(noisy(obs, noise)))
        term = u.reset_terminated.clone()
        n_crashes += int(term.sum())
        crashed |= term
        ok = ~crashed
        d = u._robot.data
        pos = d.root_pos_w.torch
        dist = torch.linalg.norm(u._goal_pos_w - pos, dim=1)
        tilt = torch.rad2deg(torch.acos((-d.projected_gravity_b.torch[:, 2]).clamp(-1, 1)))
        w = d.root_ang_vel_b.torch
        phase = (t + 1) % goal_steps
        if phase == 0 or phase > goal_steps - settle:  # settled part of each goal
            errs.append(dist[ok].cpu())
            tilts.append(tilt[ok].cpu())
            rolls.append(w[ok, 0].abs().cpu())
            # limb fidget while holding: RMS joint speed over all joints, mean |change| of each action per step
            jspd.append(d.joint_vel.torch[ok].square().mean(dim=1).sqrt().cpu())
            dact.append((u._actions - u._prev_actions)[ok].abs().mean(dim=1).cpu())
        if phase == 0:  # end of a goal: did it get there
            reached.append((dist[ok] < 0.3).float().cpu())
        angv.append(w[ok].norm(dim=1).cpu())
        if cond == "midair":  # recovery: first time tilt < 15 deg and |w| < 1 rad/s for 0.5 s straight
            calm = (tilt < 15.0) & (w.norm(dim=1) < 1.0) & ok
            calm_steps = torch.where(calm, calm_steps + 1, torch.zeros_like(calm_steps))
            just = (calm_steps >= int(0.5 / dt)) & torch.isnan(recovered_at)
            recovered_at[just] = (t + 1) * dt - 0.5
        if cond == "pushes":
            due = (t * dt >= next_push) & ok
            if bool(due.any()):
                ids = due.nonzero(as_tuple=False).squeeze(-1)
                v = root_vel(d)[ids]
                ang = torch.rand(len(ids), device=dev) * 2 * math.pi
                mag = torch.empty(len(ids), device=dev).uniform_(*args.push_dv)
                v[:, 0] += mag * torch.cos(ang)
                v[:, 1] += mag * torch.sin(ang)
                u._robot.write_root_velocity_to_sim_index(root_velocity=v, env_ids=ids)
                next_push[ids] += torch.empty(len(ids), device=dev).uniform_(0.5, 1.5) * args.push_every

    cat = lambda xs: torch.cat(xs).numpy() if xs and sum(len(x) for x in xs) else np.array([np.nan])  # noqa: E731
    e, r, ti, av, ro, js, da = cat(errs), cat(reached), cat(tilts), cat(angv), cat(rolls), cat(jspd), cat(dact)
    res = {
        "robots": N,
        "crash_rate": crashed.float().mean().item(),
        "crashes_per_robot_minute": n_crashes / N / (args.seconds / 60.0),
        "goals_reached": float(np.mean(r)),
        "hover_error_mean_m": float(np.mean(e)),
        "hover_error_p95_m": float(np.percentile(e, 95)),
        "tilt_mean_deg": float(np.mean(ti)),
        "ang_vel_mean": float(np.mean(av)),
        "roll_rate_hover_mean": float(np.mean(ro)),
        "joint_speed_hover_rms": float(np.mean(js)),
        "action_change_hover_mean": float(np.mean(da)),
    }
    if cond == "midair":
        rec = recovered_at[~torch.isnan(recovered_at)].cpu().numpy()
        res["recovered_frac"] = len(rec) / N
        res["recovery_time_median_s"] = float(np.median(rec)) if len(rec) else float("nan")
    return res


results = {}
for cond in args.conditions:
    results[cond] = run(cond)
    print(f"[EVAL] {cond}: {json.dumps({k: round(v, 4) for k, v in results[cond].items()})}")
u._max_thrust = base_thrust
u.cfg.air_start_prob = base_air_prob

labels = {
    "takeoff": "Standing take-off, random goals",
    "midair": "Mid-air start (tilt, velocity, spin)",
    "pushes": f"Velocity kicks {args.push_dv[0]}-{args.push_dv[1]} m/s every ~{args.push_every:g} s",
    "thrust_low": f"Thrust -{args.thrust_delta:.0%}",
    "thrust_high": f"Thrust +{args.thrust_delta:.0%}",
    "obs_noise": f"Observation noise (std {args.obs_noise})",
}
lines = [
    f"Checkpoint `{args.checkpoint}` on `{args.task}`, {N} robots x {args.seconds:g} s per condition, deterministic policy."
    " Crash rate = robots that crashed at least once. Hover error / tilt / roll rate = last 1.5 s of each goal,"
    " robots that hadn't crashed. Goals reached = within 0.3 m at the goal switch. Limb speed = RMS joint speed"
    " while holding (same window); action change = mean |change| of each action per 20 ms step.",
    "",
    "| condition | crash rate | goals reached | hover error mean / p95 | tilt | roll rate (hover) | limb speed (hover)"
    " | action change (hover) | recovery |",
    "|---|---|---|---|---|---|---|---|---|",
]
for cond, r in results.items():
    rec = (f"{r['recovered_frac']:.0%} in {r['recovery_time_median_s']:.2f} s (median)" if "recovered_frac" in r else "-")
    lines.append(
        f"| {labels[cond]} | {r['crash_rate']:.1%} | {r['goals_reached']:.0%} | {100 * r['hover_error_mean_m']:.1f} / "
        f"{100 * r['hover_error_p95_m']:.1f} cm | {r['tilt_mean_deg']:.1f} deg | {r['roll_rate_hover_mean']:.2f} rad/s | "
        f"{r['joint_speed_hover_rms']:.2f} rad/s | {r['action_change_hover_mean']:.3f} | {rec} |"
    )
table = "\n".join(lines)
print(table)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
meta = {"task": args.task, "checkpoint": args.checkpoint, "num_envs": N, "seconds": args.seconds, "seed": args.seed,
        "env_overrides": args.env, "push_dv": list(args.push_dv), "push_every": args.push_every,
        "thrust_delta": args.thrust_delta, "obs_noise": args.obs_noise}
with open(f"{args.out}.json", "w") as f:
    json.dump({**meta, "results": results}, f, indent=2)
with open(f"{args.out}.md", "w") as f:
    f.write(table + "\n")
print(f"[EVAL] wrote {args.out}.json and {args.out}.md")

if args.wandb:
    import wandb

    run_ = wandb.init(project="humanoid_fly", job_type="eval",
                      name=f"eval-{args.task}-{os.path.basename(args.checkpoint)[:-3]}", config=meta)
    keys = sorted({k for r in results.values() for k in r})
    cols = ["condition"] + keys
    tbl = wandb.Table(columns=cols)
    for cond, r in results.items():
        tbl.add_data(cond, *[r.get(c, float("nan")) for c in cols[1:]])
    wandb.log({"eval/table": tbl, **{f"eval/{c}/{k}": v for c, r in results.items() for k, v in r.items()}})
    run_.finish()
env.close()
app.close()
