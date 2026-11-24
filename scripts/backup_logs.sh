#!/usr/bin/env bash
# Back up logs/ (every checkpoint, TensorBoard log and training clip, plus the local wandb/ cache) as a GitHub release
# of jatinsikka/humanoid-fly, so nothing is lost when the GPU instance is shut down. Run on the GPU box, from anywhere:
#
#   bash scripts/backup_logs.sh                       # tag raw-logs-YYYY-MM-DD
#   bash scripts/backup_logs.sh raw-logs-ironman-v2   # custom tag
#
# Restore: download every humanoid_fly_logs.tar.gz.part-* asset, then in the repo root:
#   cat humanoid_fly_logs.tar.gz.part-* | tar -xzf -
set -euo pipefail
cd "$(dirname "$0")/.."

tag="${1:-raw-logs-$(date +%Y-%m-%d)}"
command -v gh >/dev/null || { echo "GitHub CLI (gh) not found: install it, then gh auth login"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "gh is not logged in: run gh auth login"; exit 1; }
[ -d logs ] || { echo "no logs/ folder in $(pwd)"; exit 1; }
if gh release view "$tag" >/dev/null 2>&1; then echo "release $tag already exists: pass another tag"; exit 1; fi

dirs=(logs)
[ -d wandb ] && dirs+=(wandb)
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
echo "packing ${dirs[*]} ($(du -shc "${dirs[@]}" | tail -1 | cut -f1)) ..."
tar -czf - "${dirs[@]}" | split -b 1900m -d - "$out/humanoid_fly_logs.tar.gz.part-"  # release assets max 2 GiB each
ls -lh "$out"

gh release create "$tag" "$out"/humanoid_fly_logs.tar.gz.part-* \
  --title "Raw training logs and checkpoints ($tag)" \
  --notes "Full ${dirs[*]} (every checkpoint, TensorBoard log and training clip of every run), archived from the GPU box.
Restore in the repo root: download all humanoid_fly_logs.tar.gz.part-* assets, then
cat humanoid_fly_logs.tar.gz.part-* | tar -xzf -"
echo "backed up to $(gh release view "$tag" --json url -q .url)"
