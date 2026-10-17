#!/bin/bash
#SBATCH --job-name=v3_profile
#SBATCH --output=logs/v3/slurm_profile_%j.log
#SBATCH --error=logs/v3/slurm_profile_%j.log
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia:1
#SBATCH -n 8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
# v3 run B2: PASTA head profiling, ONE model per job (submit twice, see below).
# Every head on its own is steered toward the identity span (alpha 0.01 on all
# other tokens) on 48 split-A items, for the 4 regions, their length-matched
# placebos and their place-name controls, under prefix, infix and target.
# Expected 3-5 h for Llama, 2-4 h for Qwen on an H200.
#   sbatch --export=ALL,MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct v3_profile_job.sbatch
#   sbatch --export=ALL,MODEL_NAME=Qwen/Qwen2.5-7B-Instruct v3_profile_job.sbatch
# logs/v3/ must exist before sbatch.
set -euo pipefail

cd "$HOME/FinalThesis/D3code-main/scripts"
source "$HOME/thesis-env-v2/bin/activate"
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
: "${MODEL_NAME:?submit with --export=ALL,MODEL_NAME=...}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python gpu_check.py
python head_profiling.py --out-dir results/v3/profiling
