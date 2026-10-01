#!/bin/bash
#SBATCH --job-name=select_items_with_baseline_filter
#SBATCH --output=logs/item_selection/select_items_with_baseline_filter_%j.log
#SBATCH --error=logs/item_selection/select_items_with_baseline_filter_%j.log
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia:1
#SBATCH -n 8
#SBATCH -t 00:30:00

set -euo pipefail

PROJECT_DIR="$HOME/FinalThesis/D3code-main/scripts"
VENV_DIR="$HOME/thesis-env"

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python select_items_with_baseline_filter.py
