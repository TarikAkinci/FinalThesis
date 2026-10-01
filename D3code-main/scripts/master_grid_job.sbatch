#!/bin/bash
#SBATCH --job-name=master_grid_llama
#SBATCH --output=logs/master_grid/master_grid_llama_%j.log
#SBATCH --error=logs/master_grid/master_grid_llama_%j.log
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia:1
#SBATCH -n 8
#SBATCH -t 12:00:00

set -euo pipefail

PROJECT_DIR="$HOME/FinalThesis/D3code-main/scripts"
VENV_DIR="$HOME/thesis-env"

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

# Smoke test before committing the full run:
#   sbatch --export=ALL,MAX_ITEMS=3 -t 00:30:00 master_grid_job.sbatch
# That still does all 500 stage-1 baselines but only 3 items through the
# condition grid, so it exercises every code path in a few minutes.
python master_grid.py
