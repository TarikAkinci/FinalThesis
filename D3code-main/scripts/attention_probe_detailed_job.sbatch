#!/bin/bash
#SBATCH --job-name=attention_probe_detailed
#SBATCH --output=logs/attention_probing/attention_probe_detailed_%j.log
#SBATCH --error=logs/attention_probing/attention_probe_detailed_%j.log
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia:1
#SBATCH -n 8
#SBATCH -t 02:00:00

set -euo pipefail

PROJECT_DIR="$HOME/FinalThesis/D3code-main/scripts"
VENV_DIR="$HOME/thesis-env"

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python attention_probe_detailed.py
