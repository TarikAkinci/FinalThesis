#!/bin/bash
#SBATCH --job-name=v2_select
#SBATCH --output=logs/v2/slurm_select_%j.log
#SBATCH --error=logs/v2/slurm_select_%j.log
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia:1
#SBATCH -n 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
# v2 run A on COMA. Scores every D3CODE item with both models, then writes the shared item list items/items_v2.csv.
# Same stages as on vast.ai (vast/run_pipeline.sh works on any machine).
# logs/v2/ must exist before sbatch, or the job vanishes without a log.
set -euo pipefail

cd "$HOME/FinalThesis/D3code-main/scripts"
source "$HOME/thesis-env-v2/bin/activate"
export HF_HOME="$HOME/.cache/huggingface"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python gpu_check.py
bash vast/run_pipeline.sh check score select
