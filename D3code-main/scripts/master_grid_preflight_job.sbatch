#!/bin/bash
#SBATCH --job-name=master_grid_preflight
#SBATCH --output=logs/master_grid/master_grid_preflight_%j.log
#SBATCH --error=logs/master_grid/master_grid_preflight_%j.log
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH -n 4
#SBATCH -t 00:30:00

# CPU-only early warning, no GPU needed. For BOTH models it checks: python imports,
# data files, item ranking, tokenizer loading from the offline HF cache, exact token
# matching, and span-finding over the whole run plan under each model's chat template.
# It exits (PREFLIGHT_ONLY=1) before any model weights are loaded. If this passes,
# the remaining failure modes are GPU-side only.

set -euo pipefail

PROJECT_DIR="$HOME/FinalThesis/D3code-main/scripts"
VENV_DIR="$HOME/thesis-env"

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PREFLIGHT_ONLY=1

echo "=== Llama ==="
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" python master_grid.py
echo
echo "=== Qwen ==="
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" python master_grid.py
echo
echo "ALL PREFLIGHTS PASSED"
