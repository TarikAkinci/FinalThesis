"""Fail fast if torch cannot run real bf16 kernels on this GPU (the LRZ V100
failure: 'no kernel image is available'), and print the library versions."""
import platform

import torch
import transformers

assert torch.cuda.is_available(), "torch sees no CUDA device"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
emb = torch.nn.Embedding(10, 64).cuda().to(torch.bfloat16)
x = emb(torch.arange(10, device="cuda"))
(x @ x.T).float().sum().item()
print(f"GPU OK: {name}, compute capability {cap[0]}.{cap[1]}, "
      f"python {platform.python_version()}, torch {torch.__version__}, transformers {transformers.__version__}")
