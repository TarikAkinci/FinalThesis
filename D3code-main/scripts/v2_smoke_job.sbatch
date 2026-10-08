#!/bin/bash
#SBATCH --job-name=v2_smoke
#SBATCH --output=logs/v2/slurm_smoke_%j.log
#SBATCH --error=logs/v2/slurm_smoke_%j.log
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia:1
#SBATCH -n 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
# v2 run A on COMA. Tokenizer preflight on every item, then 3 items through the full grid per model.
# Same stages as on vast.ai (vast/run_pipeline.sh works on any machine).
# logs/v2/ must exist before sbatch, or the job vanishes without a log.
set -euo pipefail

cd "$HOME/FinalThesis/D3code-main/scripts"
source "$HOME/thesis-env-v2/bin/activate"
export HF_HOME="$HOME/.cache/huggingface"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python gpu_check.py
bash vast/run_pipeline.sh preflight smoke
