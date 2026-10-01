#!/bin/bash
#SBATCH --job-name=download_models
#SBATCH --output=logs/master_grid/download_models_%j.log
#SBATCH --error=logs/master_grid/download_models_%j.log
#SBATCH -p lrz-cpu
#SBATCH -n 4
#SBATCH -t 01:00:00

# One-time setup: populate the HF cache on THIS cluster's filesystem (it does
# not share storage with ginkgo/COMA). Run this once, successfully, before
# any master_grid job -- those jobs run with HF_HUB_OFFLINE=1 and will fail
# outright if the weights aren't already cached.
#
# Uses a CPU partition since this is a network download, not compute -- no
# reason to occupy a GPU node for it. If lrz-cpu compute nodes turn out to
# have no outbound internet (unverified -- COMA's compute nodes didn't),
# this will fail with a connection error, and the fallback is to run the
# same huggingface-cli commands directly on the login node instead (it's a
# few GB of network I/O, not the "heavy computation" the login-node policy
# is about).
#
# Needs a HF token for Llama-3.1-8B-Instruct (gated model) -- either
# `huggingface-cli login` once interactively beforehand, or export
# HF_TOKEN=... before submitting this job.

set -euo pipefail

PROJECT_DIR="$HOME/FinalThesis/D3code-main/scripts"
VENV_DIR="$HOME/thesis-env"

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

export HF_HOME="$HOME/.cache/huggingface"
# deliberately NOT setting HF_HUB_OFFLINE here -- this job's whole purpose is
# to go online and fetch the weights

echo "Downloading meta-llama/Llama-3.1-8B-Instruct..."
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct

echo "Downloading Qwen/Qwen2.5-7B-Instruct..."
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

echo "Done. Cached under $HF_HOME"
du -sh "$HF_HOME"
