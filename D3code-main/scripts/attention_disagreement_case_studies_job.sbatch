#!/bin/bash
#SBATCH --job-name=attention_disagreement_case_studies
#SBATCH --output=logs/disagreement_case_studies/attention_disagreement_case_studies_%j.log
#SBATCH --error=logs/disagreement_case_studies/attention_disagreement_case_studies_%j.log
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia:1
#SBATCH -n 8
#SBATCH -t 01:00:00

set -euo pipefail

PROJECT_DIR="$HOME/FinalThesis/D3code-main/scripts"
VENV_DIR="$HOME/thesis-env"

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python attention_disagreement_case_studies.py
