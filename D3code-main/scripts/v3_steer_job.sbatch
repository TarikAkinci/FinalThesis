#!/bin/bash
#SBATCH --job-name=v3_steer
#SBATCH --output=logs/v3/slurm_steer_%j.log
#SBATCH --error=logs/v3/slurm_steer_%j.log
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia:1
#SBATCH -n 8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
# v3 run C: steering on the held-out half (split B), ONE model per job:
#   sbatch --export=ALL,MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct v3_steer_job.sbatch
#   sbatch --export=ALL,MODEL_NAME=Qwen/Qwen2.5-7B-Instruct v3_steer_job.sbatch
# Heads come from the frozen split-A selection (results/v3/analysis/heads_<MODEL>_*.csv,
# made by head_selection_v3.py before any split-B run). Steps, about 3-4 h per model:
#   1. attention steering toward the identity span, DAMPENER heads   (arms A-C)
#   2. same with the AMPLIFIER heads
#   3. head profiling toward the item TEXT on split A (prefix), then
#      steering toward the text with its top 10 heads on split B      (arm D)
#   4. activation steering (difference of means), the comparison method
#   5. upper bounds: attention to the region phrase removed in ALL heads,
#      at the answer position only, then at all positions
# Every steering run: selected heads + 3 random head sets with the same layer mix,
# strengths 0 (removed) to 100 (PASTA default), prefix / infix / target, with
# placebo and place-name prompts steered the same way.
# logs/v3/ must exist before sbatch.
set -euo pipefail

cd "$HOME/FinalThesis/D3code-main/scripts"
source "$HOME/thesis-env-v2/bin/activate"
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
: "${MODEL_NAME:?submit with --export=ALL,MODEL_NAME=...}"
TAG=$(basename "$MODEL_NAME" | tr . -)
SEL=results/v3/analysis
for f in "$SEL/heads_${TAG}_dampeners.csv" "$SEL/heads_${TAG}_amplifiers.csv"; do
    [ -f "$f" ] || { echo "missing $f (copy the frozen head files to COMA first)"; exit 1; }
done

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python gpu_check.py

echo "===== 1. identity, dampeners ($(date '+%T')) ====="
python steer_eval.py --target identity --heads-file "$SEL/heads_${TAG}_dampeners.csv" --top-k 99 \
    --label attention_identity_dampeners
echo "===== 2. identity, amplifiers ($(date '+%T')) ====="
python steer_eval.py --target identity --heads-file "$SEL/heads_${TAG}_amplifiers.csv" --top-k 99 \
    --label attention_identity_amplifiers
echo "===== 3. text: profiling on split A, then steering on split B ($(date '+%T')) ====="
python head_profiling.py --span text --variants prefix --out-dir results/v3/profiling
python steer_eval.py --target text --heads-file "results/v3/profiling/profiling_text_heads_${TAG}_prefix.csv" \
    --top-k 10 --label attention_text
echo "===== 4. activation steering ($(date '+%T')) ====="
python steer_eval.py --method actadd --coefs=-1,-0.5,0,0.5,1
echo "===== 5. upper bounds ($(date '+%T')) ====="
python ablation_heads.py --heads all --alphas 0 --scope last --out-dir results/v3/ablation
python ablation_heads.py --heads all --alphas 0 --scope all --out-dir results/v3/ablation
echo "===== done ($(date '+%T')) ====="
