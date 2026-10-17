#!/bin/bash
#SBATCH --job-name=v3_place
#SBATCH --output=logs/v3/slurm_place_%j.log
#SBATCH --error=logs/v3/slurm_place_%j.log
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia:1
#SBATCH -n 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
# v3 run B1: the 8 place-name controls ("someone researching / mapping <region>")
# under all 10 variants on the same 500 items, both models one after the other on
# the same GPU. Own output folder (results/v3/place, suffix _place), so the v2
# grid files stay untouched. About 15 min per model on an H200.
# logs/v3/ must exist before sbatch.
set -euo pipefail

cd "$HOME/FinalThesis/D3code-main/scripts"
source "$HOME/thesis-env-v2/bin/activate"
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python gpu_check.py
PLACES=place_researching_western_europe,place_mapping_western_europe,place_researching_latin_america,place_mapping_latin_america,place_researching_arab_culture,place_mapping_arab_culture,place_researching_sub_saharan_africa,place_mapping_sub_saharan_africa
for m in meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct; do
    MODEL_NAME=$m ONLY_CONDITIONS=$PLACES OUT_SUFFIX=_place RESULTS_DIR=results/v3/place python master_grid.py
done
