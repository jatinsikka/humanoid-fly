"""Training curves (reward, losses, episode length / crashes, action noise, reward terms) from RSL-RL TensorBoard logs.

Plain Python + numpy + matplotlib (no TensorFlow / Isaac Lab needed), so it also runs on a laptop.
Pass the run folders of one training in order (stage 1, stage 2, ...); resumed stages continue the iteration count,
and the plots mark where each stage starts.

    python scripts/plot_curves.py logs/r1_ironman_v1/<run> logs/r1_ironman_v1_smooth/<run> --out media/curves/ironman
    python scripts/plot_curves.py logs/r1_jet/<run> --out media/curves/jet --title "Jet pack"

Writes <out>_training.png (2x2 overview), <out>_reward_terms.png and <out>.csv.
"""

import argparse
import csv
import glob
import os
import struct
from collections import defaultdict

import numpy as np


# ------------------------------------------------------------------ minimal TensorBoard event reader
def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def _fields(b):
    i = 0
    while i < len(b):
        k, i = _varint(b, i)
        f, t = k >> 3, k & 7
        if t == 0:
            v, i = _varint(b, i)
        elif t == 1:
            v, i = b[i:i + 8], i + 8
        elif t == 5:
            v, i = b[i:i + 4], i + 4
        elif t == 2:
            n, i = _varint(b, i)
            v, i = b[i:i + n], i + n
        else:
            raise ValueError(f"wire type {t}")
        yield f, v


def read_scalars(run_dir):
    """{tag: {step: value}} from every events.out.tfevents.* file in run_dir."""
    out = defaultdict(dict)
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not files:
        raise SystemExit(f"no TensorBoard event files in {run_dir}")
    for path in files:
        data = open(path, "rb").read()
        i = 0
        while i + 12 <= len(data):
            n = struct.unpack("<Q", data[i:i + 8])[0]
            ev = data[i + 12:i + 12 + n]
            i += 12 + n + 4
            step, summary = 0, None
            for f, v in _fields(ev):
                if f == 2:
                    step = v
                elif f == 5:
                    summary = v
            if summary is None:
                continue
            for f, v in _fields(summary):
                if f != 1:
                    continue
                tag = val = None
                for g, w in _fields(v):
                    if g == 1:
                        tag = w.decode()
                    elif g == 2:
                        val = struct.unpack("<f", w)[0]
                if tag is not None and val is not None:
                    out[tag][step] = val
    return out


def merge_runs(run_dirs):
    """Concatenate stages into one iteration axis.

    A resumed stage (RSL-RL continues the iteration count from the loaded checkpoint, e.g. 1300) replaces whatever
    the previous stage logged from that iteration on, since training continued from that checkpoint. A stage that
    starts again at 0 (not resumed) is shifted to follow the previous one.
    """
    merged, starts, end = defaultdict(dict), [], 0
    for d in run_dirs:
        s = read_scalars(d)
        steps = sorted(s.get("Train/mean_reward", next(iter(s.values()))).keys())
        if not steps:
            continue
        first = steps[0]
        if merged and first == 0:
            shift = end
        else:
            shift = 0
            for tag in merged:  # drop the previous stage's iterations past the resume point
                merged[tag] = {k: v for k, v in merged[tag].items() if k < first}
        starts.append((first + shift, os.path.basename(os.path.dirname(d.rstrip("/"))) or d))
        for tag, ser in s.items():
            for k, v in ser.items():
                merged[tag][k + shift] = v
        end = steps[-1] + shift + 1
    return merged, starts


def series(m, tag):
    if tag not in m or not m[tag]:
        return np.array([]), np.array([])
    k = np.array(sorted(m[tag]))
    return k, np.array([m[tag][i] for i in k])


def ema(y, alpha):
    if alpha <= 0 or len(y) == 0:
        return y
    out = np.empty_like(y)
    acc = y[0]
    for i, v in enumerate(y):
        acc = alpha * acc + (1 - alpha) * v
        out[i] = acc
    return out


# ------------------------------------------------------------------ plots
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders (containing events.out.tfevents.*), in training order")
    ap.add_argument("--out", required=True, help="output prefix, e.g. media/curves/ironman")
    ap.add_argument("--title", default=None)
    ap.add_argument("--wandb", action="store_true", help="also upload the plots to W&B (project humanoid_fly)")
    ap.add_argument("--smooth", type=float, default=0.9, help="EMA factor for the bold line (raw shown faint); 0 = off")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m, starts = merge_runs(args.runs)
    title = args.title or os.path.basename(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def plot(ax, tag, label, color, scale=1.0):
        x, y = series(m, tag)
        if len(x) == 0:
            return False
        y = y * scale
        if args.smooth > 0:
            ax.plot(x, y, color=color, alpha=0.25, lw=0.8)
        ax.plot(x, ema(y, args.smooth), color=color, lw=1.8, label=label)
        return True

    def stages(ax):
        for i, (x0, name) in enumerate(starts):
            if i > 0:
                ax.axvline(x0, color="0.5", ls="--", lw=0.8)
        ax.grid(alpha=0.3)
        ax.set_xlabel("PPO iteration")

    fig, axs = plt.subplots(2, 2, figsize=(12, 7.5))
    ax = axs[0, 0]
    plot(ax, "Train/mean_reward", "mean episode reward", "C0")
    ax.set_title("Reward")
    stages(ax)
    ax = axs[0, 1]
    plot(ax, "Loss/value", "value loss", "C1")
    ax.set_yscale("log")
    ax2 = ax.twinx()
    x, y = series(m, "Loss/surrogate")
    if len(x):
        ax2.plot(x, ema(y, args.smooth), color="C2", lw=1.4, label="surrogate (policy) loss")
        ax2.set_ylabel("surrogate loss", color="C2")
    ax.set_ylabel("value loss (log)", color="C1")
    ax.set_title("PPO losses")
    stages(ax)
    ax = axs[1, 0]
    plot(ax, "Train/mean_episode_length", "episode length [s]", "C3", scale=1 / 50.0)  # 50 policy steps per s
    ax.set_ylabel("mean episode length [s]")
    ax2 = ax.twinx()
    x, y = series(m, "Episode_Termination/died")
    if len(x):
        ax2.plot(x, ema(y, args.smooth), color="C4", lw=1.4)
        ax2.set_ylabel("crashes per reset batch", color="C4")
    ax.set_title("Survival: episode length (max 12 s) and crashes")
    stages(ax)
    ax = axs[1, 1]
    plot(ax, "Metrics/final_distance_to_goal", "final distance to goal [m]", "C5")
    ax.set_ylabel("final distance to goal [m]")
    ax2 = ax.twinx()
    x, y = series(m, "Policy/mean_std")
    if len(x):
        ax2.plot(x, y, color="C6", lw=1.2)
        ax2.set_ylabel("action noise std", color="C6")
    ax.set_title("Accuracy and exploration")
    stages(ax)
    fig.suptitle(f"{title}: training curves" + ("" if len(starts) < 2 else f"  ({len(starts)} stages, dashed = stage start)"))
    fig.tight_layout()
    fig.savefig(f"{args.out}_training.png", dpi=130)
    plt.close(fig)

    terms = sorted(t for t in m if t.startswith("Episode_Reward/"))
    if terms:
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, t in enumerate(terms):
            x, y = series(m, t)
            ax.plot(x, ema(y, args.smooth), lw=1.4, label=t.split("/", 1)[1], color=plt.cm.tab20(i % 20))
        ax.axhline(0, color="k", lw=0.6)
        stages(ax)
        ax.set_ylabel("per-second reward contribution")
        ax.set_title(f"{title}: reward terms (positive = rewards, negative = penalties)")
        ax.legend(ncol=4, fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(f"{args.out}_reward_terms.png", dpi=130)
        plt.close(fig)

    tags = sorted(m)
    steps = sorted(set().union(*[m[t].keys() for t in tags]))
    with open(f"{args.out}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration"] + tags)
        for s in steps:
            w.writerow([s] + [m[t].get(s, "") for t in tags])
    if args.wandb:
        import wandb

        run = wandb.init(project="humanoid_fly", job_type="curves", name=f"curves-{title}", config={"runs": args.runs})
        imgs = {"curves/training": wandb.Image(f"{args.out}_training.png")}
        if terms:
            imgs["curves/reward_terms"] = wandb.Image(f"{args.out}_reward_terms.png")
        wandb.log(imgs)
        run.finish()
    print(f"wrote {args.out}_training.png, {args.out}_reward_terms.png, {args.out}.csv "
          f"({len(steps)} iterations, stages start at {[s for s, _ in starts]})")


if __name__ == "__main__":
    main()
