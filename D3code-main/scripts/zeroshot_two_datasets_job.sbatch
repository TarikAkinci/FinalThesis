#!/bin/bash
#SBATCH --job-name=zeroshot_two_datasets
#SBATCH --output=logs/zeroshot/zeroshot_two_datasets_%j.log
#SBATCH --error=logs/zeroshot/zeroshot_two_datasets_%j.log
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia:1
#SBATCH -n 8
#SBATCH -t 05:00:00

set -euo pipefail

# --- adjust these two paths to match your actual layout on the cluster ---
PROJECT_DIR="$HOME/FinalThesis/D3code-main/scripts"
VENV_DIR="$HOME/thesis-env"
# ---------------------------------------------------------------------

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

# model + MHS dataset should already be cached (see pre-download step) so the
# compute node doesn't need outbound internet access
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# leave MODEL_NAMES unset to run the default list (Llama-3.1-8B-Instruct,
# Qwen2.5-7B-Instruct) sequentially in this one job; override with a
# comma-separated list to run a different set, e.g.:
#   MODEL_NAMES=Qwen/Qwen3-8B sbatch zeroshot_two_datasets_job.sbatch
python zeroshot_two_datasets.py
