"""Print one line per N iterations from an rsl_rl console log: python3 scripts/summarize_log.py LOG [every]"""
import re, sys
every = int(sys.argv[2]) if len(sys.argv) > 2 else 50
keys = ["Mean reward", "Mean episode length", "Mean action std", "Metrics/final_distance_to_goal",
        "Episode_Termination/died", "Episode_Reward/ang_vel", "Episode_Reward/upright", "Episode_Reward/pos_fine"]
short = ["rew", "len", "std", "dist", "died", "angv", "upr", "fine"]
blocks = re.split(r"Learning iteration ", open(sys.argv[1], errors="ignore").read())[1:]
for b in blocks:
    it = int(b.split("/")[0])
    if it % every and b is not blocks[-1]:
        continue
    vals = []
    for k, s in zip(keys, short):
        m = re.search(re.escape(k) + r":\s*(-?[\d.]+)", b)
        vals.append(f"{s}={float(m.group(1)):.3g}" if m else f"{s}=?")
    print(it, " ".join(vals))
