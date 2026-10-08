#!/bin/bash
# Staged pipeline for one single-GPU box (vast.ai). Run inside tmux so a dropped
# SSH connection does not kill it:   tmux new -s grid
#
#   bash vast/run_pipeline.sh                  # run A: check score select preflight smoke grid reproduce bundle
#   bash vast/run_pipeline.sh smoke            # any subset, in the order given
#
# Stages
#   check      tokenizer-only: identical span lengths across models, exact placebos
#   score      baseline P(yes) of every D3CODE item, per model     -> results/$RUN/scores
#   select     one shared item list (in band for both models, N, A/B split) -> items/items_$RUN.csv
#              skipped if that file exists (it is frozen once committed; FORCE=1 redoes it)
#   preflight  every prompt of the grid, every item, both models (no weights)
#   smoke      3 items through the full grid per model         -> results/$RUN/master_grid_smoke
#   grid       full grid per model (skipped if its CSV exists; FORCE=1 reruns)
#   reproduce  v2 vs LRZ (v1) numbers on identical prompts
#   bundle     one tarball with everything to copy home        -> ~/bundle_$RUN.tgz
#
# Run B stages (after the run-A analysis froze the candidate heads):
#   profile    PASTA head profiling on split-A items, one head at a time -> results/$RUN/profiling
#   ablation   candidate heads vs 10 random sets on split-B items      -> results/$RUN/ablation
#              needs results/$RUN/analysis/perhead_candidates_<MODEL>.csv; extra options
#              via ABLATION_ARGS, e.g. ABLATION_ARGS="--heads all --alphas 0"
#
# Everything goes to results/$RUN, figures/$RUN, logs/$RUN (RUN defaults to v2), so
# the LRZ results in results/master_grid stay untouched.
set -euo pipefail
cd "$(dirname "$0")/.."            # D3code-main/scripts

RUN=${RUN:-v2}
RES=results/$RUN/master_grid
SCORES=results/$RUN/scores
ITEMS=items/items_$RUN.csv
FIG=figures/$RUN
LOG=logs/$RUN
MODELS=(${MODELS:-meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct})
mkdir -p "$LOG"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export ITEMS_CSV=$ITEMS SCORES_DIR=$SCORES

tag() { basename "$1" | tr . -; }      # same as master_grid.py's MODEL_TAG default

stage_check() {
    python check_matching.py --models "${MODELS[@]}" 2>&1 | tee "$LOG/check_matching.log"
}

stage_score() {
    for m in "${MODELS[@]}"; do
        if [ -f "$SCORES/scores_$(tag "$m").csv" ] && [ "${FORCE:-0}" != 1 ]; then
            echo "skip score for $m: exists"; continue
        fi
        MODEL_NAME=$m python select_items.py score --out "$SCORES" 2>&1 | tee "$LOG/score_$(tag "$m").log"
    done
}

stage_select() {
    if [ -f "$ITEMS" ] && [ "${FORCE:-0}" != 1 ]; then
        echo "skip select: $ITEMS exists (frozen; FORCE=1 to redo)"; return
    fi
    python select_items.py select --scores "$SCORES" --out "$ITEMS" 2>&1 | tee "$LOG/select.log"
}

stage_preflight() {
    for m in "${MODELS[@]}"; do
        MODEL_NAME=$m PREFLIGHT_ONLY=1 RESULTS_DIR=$RES python master_grid.py 2>&1 \
            | tee "$LOG/preflight_$(tag "$m").log"
    done
}

stage_smoke() {
    for m in "${MODELS[@]}"; do
        MODEL_NAME=$m MAX_ITEMS=3 NOISE_ITEMS=1 RESULTS_DIR=$RES FIGURES_DIR=$FIG/smoke python master_grid.py 2>&1 \
            | tee "$LOG/smoke_$(tag "$m").log"
    done
}

stage_grid() {
    for m in "${MODELS[@]}"; do
        if [ -f "$RES/master_grid_$(tag "$m").csv.gz" ] && [ "${FORCE:-0}" != 1 ]; then
            echo "skip grid for $m: $RES/master_grid_$(tag "$m").csv.gz exists (FORCE=1 to rerun)"
            continue
        fi
        MODEL_NAME=$m RESULTS_DIR=$RES FIGURES_DIR=$FIG python master_grid.py 2>&1 | tee "$LOG/grid_$(tag "$m").log"
    done
}

stage_reproduce() {
    python check_reproduction.py --v1 results/master_grid --v2 "results/$RUN" 2>&1 | tee "$LOG/reproduce.log"
}

stage_profile() {
    for m in "${MODELS[@]}"; do
        MODEL_NAME=$m python head_profiling.py --out-dir "results/$RUN/profiling" ${PROFILE_ARGS:-} 2>&1 \
            | tee "$LOG/profile_$(tag "$m").log"
    done
}

stage_ablation() {
    for m in "${MODELS[@]}"; do
        t=$(tag "$m")
        MODEL_NAME=$m python ablation_heads.py --candidates "results/$RUN/analysis/perhead_candidates_$t.csv" \
            --out-dir "results/$RUN/ablation" ${ABLATION_ARGS:-} 2>&1 | tee "$LOG/ablation_$t.log"
    done
}

stage_bundle() {
    python - <<'EOF' > "$LOG/env_versions.txt"
import platform, torch, transformers, pandas, numpy, scipy
print("python", platform.python_version())
for m in (torch, transformers, pandas, numpy, scipy):
    print(m.__name__, m.__version__)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
EOF
    tar czf ~/bundle_$RUN.tgz "results/$RUN" "$FIG" "$LOG" "$ITEMS"
    ls -lh ~/bundle_$RUN.tgz
    echo "copy home (on your laptop):  scp -P <PORT> root@<IP>:~/bundle_$RUN.tgz ~/Desktop/FinalThesis/D3code-main/scripts/"
}

stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(check score select preflight smoke grid reproduce bundle)
for s in "${stages[@]}"; do
    echo "===== stage: $s  ($(date '+%F %T')) ====="
    "stage_$s"
done
echo "===== done ($(date '+%F %T')) ====="
