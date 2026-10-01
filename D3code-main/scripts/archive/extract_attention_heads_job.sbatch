#!/bin/bash
#SBATCH --job-name=extract_attention_heads
#SBATCH --output=logs/extract_attention_heads_%j.log
#SBATCH --error=logs/extract_attention_heads_%j.log
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia:1
#SBATCH -n 8
#SBATCH -t 02:00:00

set -euo pipefail

# --- adjust these two paths to match your actual layout on the cluster ---
PROJECT_DIR="$HOME/FinalThesis/D3code-main/scripts"
VENV_DIR="$HOME/thesis-env"
# ---------------------------------------------------------------------

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

# model is already cached from earlier runs, so the compute node doesn't
# need outbound internet access
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python extract_attention_heads.py
