"""
Inject the winning DiffMean direction (from probe_attention_heads.py) back
into its (layer, head) during real classification, using nnsight, and check
whether it shifts predictions on high-disagreement items specifically.

Why nnsight here and not another hand-written hook: this is a WRITE on
activations, not a read -- more failure-prone than the extraction step (we
already hit a real tuple-vs-tensor bug hand-writing an intervention hook
earlier). nnsight's trace/setting API (model.<module>.output[:] = ...,
confirmed against the official README) handles the tuple reconstruction for
us. The one thing NOT yet confirmed against docs is the exact indexing depth
of `.input` (only `.output` had a literal example) -- so this script prints
and asserts the captured shape before trusting the edit below it. If the
assert fails, the printed shape tells you how to fix the indexing rather than
silently running on wrong data.

Needs: pip install nnsight   (add to thesis-env on the cluster)
"""
import numpy as np
import pandas as pd
import torch
from nnsight import LanguageModel

from zeroshot_two_datasets import PROMPT
from extract_activations import prepare_items

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DIRECTIONS_PATH = "attention_head_directions.npz"
RESULTS_PATH = "attention_head_diffmean_results.csv"
ALPHA = 8.0    # steering strength on the unit-normalized direction
N_ITEMS = 20   # quick check on the highest-disagreement test items, not a full re-evaluation


def build_prompt(tok, text):
    messages = [{"role": "user", "content": PROMPT.format(text=text)}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


if __name__ == "__main__":
    results = pd.read_csv(RESULTS_PATH).sort_values("test_auc", ascending=False)
    best = results.iloc[0]
    layer, head = int(best["layer"]), int(best["head"])
    print(f"Steering at layer={layer}, head={head} (test AUC={best['test_auc']:.3f})")

    dirs = np.load(DIRECTIONS_PATH)
    li = list(dirs["layers"]).index(layer)
    direction = dirs["directions"][li, head]
    direction = direction / np.linalg.norm(direction)  # unit vector; ALPHA controls magnitude separately

    print(f"Loading model: {MODEL_NAME}")
    model = LanguageModel(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16, dispatch=True)
    tok = model.tokenizer
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    df = prepare_items()
    df_test = df[df["split"] == "test"].sort_values("entropy", ascending=False).head(N_ITEMS)

    # --- verification step: confirm .input's shape/indexing BEFORE trusting the edit ---
    probe_prompt = build_prompt(tok, df_test.iloc[0]["text"])
    with model.trace(probe_prompt):
        o_proj_input = model.model.layers[layer].self_attn.o_proj.input[0][0].save()
    print(f"o_proj input shape: {tuple(o_proj_input.shape)} (expect [1, seq_len, {model.config.hidden_size}])")
    assert o_proj_input.shape[-1] == model.config.hidden_size, \
        "unexpected shape -- .input indexing depth is likely different than assumed here; fix before continuing"

    direction_t = torch.tensor(direction, dtype=torch.bfloat16)
    zero_id = tok.encode("0", add_special_tokens=False)[0]
    one_id = tok.encode("1", add_special_tokens=False)[0]

    print(f"\n{'entropy':>8} {'label':>5} {'base':>5} {'steered':>7}")
    flips = 0
    for _, row in df_test.iterrows():
        prompt = build_prompt(tok, row["text"])

        with model.trace(prompt):
            baseline_logits = model.output.logits[0, -1].save()

        with model.trace(prompt):
            o_proj_in = model.model.layers[layer].self_attn.o_proj.input[0][0]
            o_proj_in[:, -1, head * head_dim:(head + 1) * head_dim] += ALPHA * direction_t.to(o_proj_in.device)
            steered_logits = model.output.logits[0, -1].save()

        base_pred = int(baseline_logits[one_id] > baseline_logits[zero_id])
        steer_pred = int(steered_logits[one_id] > steered_logits[zero_id])
        flipped = steer_pred != base_pred
        flips += flipped
        print(f"{row['entropy']:>8.2f} {row['label']:>5} {base_pred:>5} {steer_pred:>7}{'  <-- FLIPPED' if flipped else ''}")

    print(f"\n{flips}/{len(df_test)} predictions flipped by steering this one head at alpha={ALPHA}")
