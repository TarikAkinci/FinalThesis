#!/bin/bash
# One-time setup on a fresh vast.ai GPU instance (PyTorch template).
#
#   git clone https://github.com/TarikAkinci/FinalThesis.git && cd FinalThesis
#   hf auth login          # you paste your Hugging Face token; needs Llama 3.1 access
#   bash D3code-main/scripts/vast/setup_vast.sh
#
# Installs the Python deps (transformers pinned to the version the local tests
# ran with, so chat templates and the attention hook behave the same), checks
# that torch has working kernels for this GPU (the LRZ V100 failure), and
# downloads both models (~31 GB; Llama's duplicate original/ checkpoint skipped).
set -euo pipefail
cd "$(dirname "$0")/.."            # D3code-main/scripts

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python -c "import torch" 2>/dev/null || pip install -q torch
pip install -q "transformers==5.14.1" "huggingface_hub>=1.0" pandas scipy statsmodels matplotlib

python - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch sees no CUDA device"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
# run real kernels (embedding + bf16 matmul), the ops that failed on the V100
emb = torch.nn.Embedding(10, 64).cuda().to(torch.bfloat16)
x = emb(torch.arange(10, device="cuda"))
(x @ x.T).float().sum().item()
print(f"GPU OK: {name}, compute capability {cap[0]}.{cap[1]}, torch {torch.__version__}")
EOF

hf auth whoami >/dev/null 2>&1 || { echo "Not logged in to Hugging Face: run 'hf auth login' first." >&2; exit 1; }
hf download meta-llama/Llama-3.1-8B-Instruct --exclude "original/*"
hf download Qwen/Qwen2.5-7B-Instruct

df -h . | tail -1
echo "Setup done. Next:  tmux new -s grid   then   bash vast/run_pipeline.sh"
