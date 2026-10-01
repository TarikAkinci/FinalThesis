#!/bin/bash
#SBATCH --job-name=master_grid_qwen
#SBATCH --output=logs/master_grid/master_grid_qwen_%j.log
#SBATCH --error=logs/master_grid/master_grid_qwen_%j.log
#SBATCH -p lrz-v100x2
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH -t 12:00:00

# LRZ AI Systems version (COMA/ginkgo is down for maintenance). Differences
# from the COMA sbatch: partition is a named GPU-type partition here, not one
# shared "compute" queue (lrz-v100x2 = 2x V100/node, picked as the closest
# equivalent to COMA's single-GPU jobs; swap to lrz-hgx-h100-94x4 if this
# queue is slow -- 30 nodes there vs 4 for v100x2). gres syntax is
# --gres=gpu:1, not --gres=gpu:nvidia:1 (that :nvidia: tag was COMA-specific).
#
# This is a SEPARATE filesystem from ginkgo -- nothing carries over. Before
# this job can run: (1) ~/thesis-env must exist here (see venv setup step),
# (2) the HF cache must already contain both models (see
# download_models_job.sbatch, run ONCE first).

set -euo pipefail

PROJECT_DIR="$HOME/FinalThesis/D3code-main/scripts"
VENV_DIR="$HOME/thesis-env"

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

# Smoke test before committing the full run:
#   sbatch --export=ALL,MAX_ITEMS=3 -t 00:30:00 master_grid_job_lrz.sbatch
# That still does all 500 stage-1 baselines but only 3 items through the
# condition grid, so it exercises every code path (including whether the
# model actually loads from cache) in a few minutes.
python master_grid.py
