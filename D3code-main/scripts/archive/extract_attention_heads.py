"""
Extract per-ATTENTION-HEAD outputs (not residual-stream hidden states) from
Llama-3.1-8B, at the same layers Gurgurov et al. used [11, 13, 26, 28, 31].
Per supervisor: attention itself is the thing to probe, not the residual
stream we probed in extract_activations.py.

Why this needs a different extraction mechanism than output_hidden_states:
the residual stream is the MIXED result after a layer's 32 heads' outputs are
combined by its output projection (o_proj). output_attentions=True doesn't
give what we want either -- that's only the attention WEIGHTS (which tokens
each head looks at), not the actual content each head produces. What we want
is each head's own 128-dim output vector, BEFORE o_proj mixes all 32 of them
into one 4096-dim vector. That tensor is o_proj's *input*, so we capture it
with a forward PRE-hook on o_proj (a pre-hook receives a module's arguments
before forward() runs), then reshape the captured (batch, seq, 4096) tensor
into (batch, seq, num_heads=32, head_dim=128) to recover individual heads.

Same item/label prep and same classification prompt as extract_activations.py
(imported directly, not duplicated).

Run as a batch job (see extract_attention_heads_job.sbatch) -- needs a GPU.
"""
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from extract_activations import prepare_items, build_prompt

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
LAYERS = [11, 13, 26, 28, 31]
OUT_PATH = "attention_heads.npz"
MAX_LENGTH = 512


def register_head_hooks(model, layers):
    """One forward pre-hook per target layer's o_proj, capturing its input
    (the concatenated-but-not-yet-mixed per-head output) into `captured`.
    Returns (captured dict, list of handles to remove afterward)."""
    captured = {}

    def make_hook(layer_idx):
        def hook(module, args):
            captured[layer_idx] = args[0]  # (batch, seq, hidden_size), pre-o_proj
        return hook

    handles = [
        model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(make_hook(L))
        for L in layers
    ]
    return captured, handles


def extract(model, tok, texts, layers, num_heads, head_dim):
    """One forward pass per item. Returns [n_items, len(layers), num_heads, head_dim]
    -- each head's own output vector at the LAST TOKEN position, per layer."""
    captured, handles = register_head_hooks(model, layers)
    feats = np.zeros((len(texts), len(layers), num_heads, head_dim), dtype=np.float32)
    try:
        with torch.no_grad():
            for i, text in enumerate(texts):
                prompt = build_prompt(tok, text)
                inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
                model(**inputs)  # hidden_states/attentions not needed -- hooks capture what we want
                for j, L in enumerate(layers):
                    last_tok = captured[L][0, -1]              # (hidden_size,)
                    feats[i, j] = last_tok.view(num_heads, head_dim).float().cpu().numpy()
                if (i + 1) % 200 == 0:
                    print(f"  {i + 1}/{len(texts)}")
    finally:
        for h in handles:
            h.remove()
    return feats


if __name__ == "__main__":
    df = prepare_items()

    print(f"\nLoading model: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    print(f"num_attention_heads={num_heads}, head_dim={head_dim}")

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    print(f"\nExtracting train attention-head outputs (n={len(df_train)})...")
    X_train = extract(model, tok, df_train["text"].tolist(), LAYERS, num_heads, head_dim)
    print(f"Extracting test attention-head outputs (n={len(df_test)})...")
    X_test = extract(model, tok, df_test["text"].tolist(), LAYERS, num_heads, head_dim)

    np.savez(
        OUT_PATH,
        X_train=X_train, y_train=df_train["label"].values,
        X_test=X_test, y_test=df_test["label"].values,
        layers=np.array(LAYERS), num_heads=num_heads, head_dim=head_dim,
    )
    print(f"\nSaved to {OUT_PATH}  (shape: {X_train.shape} train, {X_test.shape} test)")
