#!/usr/bin/env bash
# bash scripts/train.sh [--num_envs N] [--max_iterations N] [--task Humanoid-Fly-R1-Jet-WholeBody-v0]
cd "$(dirname "$0")/.." && ~/IsaacLab/isaaclab.sh -p scripts/train.py --headless "$@"
