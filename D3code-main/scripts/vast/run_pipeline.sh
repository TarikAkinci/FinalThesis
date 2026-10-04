#!/bin/bash
# Staged pipeline for a single-GPU box (vast.ai). Run inside tmux so a dropped
# SSH connection does not kill it:   tmux new -s grid
#
#   bash vast/run_pipeline.sh                      # preflight smoke grid analysis
#   bash vast/run_pipeline.sh analysis             # any subset, in the order given
#   bash vast/run_pipeline.sh ablation             # after reading the candidates
#   bash vast/run_pipeline.sh bundle               # tar the npz files for download
#
# Stages
#   preflight  tokenizer-only checks for both models (no GPU time)
#   smoke      3 items through the full grid per model -> results/$RUN/master_grid_smoke
#   grid       full grid per model (skipped if its CSV exists; FORCE=1 reruns)
#   analysis   CPU analyses: master table, length control, sink, group gap,
#              per-head candidates, cross-model comparison
#   ablation   causal pilot on each model's candidate heads (ABLATION_ARGS are
#              passed through, e.g. ABLATION_ARGS="--allow-failed --alphas 0,2,4")
#   bundle     results/$RUN/perhead_npz.tgz with the per-head npz files (gitignored,
#              ~100-250 MB) to scp home -- they are lost when the instance is destroyed
#
# Everything goes to results/$RUN, figures/$RUN, logs/$RUN (RUN defaults to v2), so
# the LRZ results in results/master_grid stay untouched. What to commit afterwards:
# results/$RUN (CSV/CSV.gz/md), logs/$RUN, figures/$RUN. Not the npz.
set -euo pipefail
cd "$(dirname "$0")/.."            # D3code-main/scripts

RUN=${RUN:-v2}
RES=results/$RUN/master_grid
ANA=results/$RUN/analysis
ABL=results/$RUN/ablation
FIG=figures/$RUN
LOG=logs/$RUN
MODELS=(${MODELS:-meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct})
mkdir -p "$LOG"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

tag() { basename "$1" | tr . -; }      # same as master_grid.py's MODEL_TAG default

stage_preflight() {
    for m in "${MODELS[@]}"; do
        MODEL_NAME=$m PREFLIGHT_ONLY=1 RESULTS_DIR=$RES python master_grid.py 2>&1 | tee "$LOG/preflight_$(tag "$m").log"
    done
}

stage_smoke() {
    for m in "${MODELS[@]}"; do
        MODEL_NAME=$m MAX_ITEMS=3 RESULTS_DIR=$RES FIGURES_DIR=$FIG/smoke python master_grid.py 2>&1 \
            | tee "$LOG/smoke_$(tag "$m").log"
    done
}

stage_grid() {
    for m in "${MODELS[@]}"; do
        if [ -f "$RES/master_grid_$(tag "$m").csv" ] && [ "${FORCE:-0}" != 1 ]; then
            echo "skip grid for $m: $RES/master_grid_$(tag "$m").csv exists (FORCE=1 to rerun)"
            continue
        fi
        MODEL_NAME=$m RESULTS_DIR=$RES FIGURES_DIR=$FIG python master_grid.py 2>&1 | tee "$LOG/grid_$(tag "$m").log"
    done
}

stage_analysis() {
    mkdir -p "$ANA"
    RESULTS_DIR=$RES MASTER_TABLE_CSV=results/$RUN/master_table.csv python build_master_table.py 2>&1 | tee "$LOG/master_table.log"
    python analysis_length_control.py "$RES" "$ANA" 2>&1 | tee "$LOG/length_control.log"
    python analysis_sink_dynamics.py "$RES" "$ANA" 2>&1 | tee "$LOG/sink_dynamics.log"
    perhead=()
    for m in "${MODELS[@]}"; do
        npz=$RES/master_grid_perhead_$(tag "$m").npz
        [ -f "$npz" ] || { echo "missing $npz"; continue; }
        perhead+=(--perhead "$npz")
        python analysis_perhead_candidates.py --npz "$npz" --out-dir "$ANA" 2>&1 | tee "$LOG/perhead_$(tag "$m").log"
    done
    python analysis_group_gap.py "$RES" "$ANA" "${perhead[@]}" 2>&1 | tee "$LOG/group_gap.log"
    python analysis_perhead_compare.py "$ANA" 2>&1 | tee "$LOG/perhead_compare.log"
}

stage_ablation() {
    mkdir -p "$ABL"
    for m in "${MODELS[@]}"; do
        t=$(tag "$m")
        MODEL_NAME=$m python ablation_heads.py --candidates "$ANA/perhead_candidates_$t.csv" \
            --items "$RES/master_grid_items_$t.csv" --out-dir "$ABL" ${ABLATION_ARGS:-} 2>&1 \
            | tee "$LOG/ablation_$t.log" || echo "ablation for $m did not run (see $LOG/ablation_$t.log)"
    done
}

stage_bundle() {
    tar czf "results/$RUN/perhead_npz.tgz" "$RES"/master_grid_perhead_*.npz
    ls -lh "results/$RUN/perhead_npz.tgz"
    echo "download with:  scp -P <port> root@<host>:$(pwd)/results/$RUN/perhead_npz.tgz ."
}

stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(preflight smoke grid analysis)
for s in "${stages[@]}"; do
    echo "===== stage: $s  ($(date '+%F %T')) ====="
    "stage_$s"
done
echo "===== done ($(date '+%F %T')) ====="
