#!/bin/bash
# One-time setup on a fresh vast.ai GPU instance (use a PyTorch template/image).
#
#   git clone https://github.com/TarikAkinci/FinalThesis.git && cd FinalThesis
#   export HF_TOKEN=hf_...      # Hugging Face token of the account that has Llama 3.1 access
#   bash D3code-main/scripts/vast/setup_vast.sh
#
# Installs the Python deps, checks that torch actually has kernels for this GPU
# (the LRZ V100 failure: "no kernel image is available"), and downloads both
# models (~31 GB; the duplicate original/ checkpoint of Llama is skipped).
set -euo pipefail
cd "$(dirname "$0")/.."            # D3code-main/scripts
mkdir -p results logs

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python -c "import torch" 2>/dev/null || pip install -q torch
pip install -q "transformers>=5" pandas scipy matplotlib huggingface_hub

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

if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is not set: export it (needed for the gated Llama model) and rerun." >&2
    exit 1
fi
hf download meta-llama/Llama-3.1-8B-Instruct --exclude "original/*"
hf download Qwen/Qwen2.5-7B-Instruct

python - <<'EOF' | tee logs/env_versions.txt
import platform, torch, transformers, pandas, numpy, scipy
print("python", platform.python_version())
for m in (torch, transformers, pandas, numpy, scipy):
    print(m.__name__, m.__version__)
print("gpu", torch.cuda.get_device_name(0))
EOF
df -h . | tail -1
echo "Setup done. Next: bash vast/run_pipeline.sh"
