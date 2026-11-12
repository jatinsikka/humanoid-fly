#!/usr/bin/env bash
# bash scripts/play.sh [--checkpoint logs/r1_jet/<run>/model_N.pt] [--num_envs 4]
cd "$(dirname "$0")/.." && ~/IsaacLab/isaaclab.sh -p scripts/play.py "$@"
