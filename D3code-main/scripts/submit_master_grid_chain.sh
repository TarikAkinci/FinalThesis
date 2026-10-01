#!/bin/bash
# Queues, per model: a 3-item smoke test, then the full run that only starts if the
# smoke test exited 0 (--dependency=afterok). If a smoke test fails, its full run is
# cancelled instead of burning hours of GPU time on the same error.
# Usage: bash submit_master_grid_chain.sh [partition]     (default: lrz-hgx-h100-94x4)
set -euo pipefail
cd "$HOME/FinalThesis/D3code-main/scripts"
P="${1:-lrz-hgx-h100-94x4}"
echo "Partition: $P"

for pair in "llama:master_grid_job_lrz.sbatch" "qwen:master_grid_qwen_job_lrz.sbatch"; do
    name="${pair%%:*}"; script="${pair##*:}"
    smoke=$(sbatch --parsable -p "$P" --export=ALL,MAX_ITEMS=3 -t 00:30:00 \
            --job-name="smoke_${name}" "$script")
    full=$(sbatch --parsable -p "$P" --export=ALL,MAX_ITEMS=0 \
           --dependency="afterok:${smoke}" --kill-on-invalid-dep=yes "$script")
    echo "$name: smoke=$smoke  full=$full (starts only if smoke succeeds)"
done
echo
squeue -u "$USER"
