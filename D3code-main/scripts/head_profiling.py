"""
PASTA-style head profiling (Zhang et al. 2024): steer ONE head at a time
toward the identity span and measure what it does to the answer. Causal head
selection, as a cross-check on the correlational candidates from
analysis_perhead_candidates.py. GPU.

Steering: in the chosen head, at every query position, attention to tokens
outside the identity span is scaled by ALPHA_PASTA (0.01) and the row
renormalized -- the same as multiplying attention to the span by 1/0.01. Uses
the attention hook from ablation_heads.py (scope=all, alpha=100).

Per head, on selection-half (split A) items only, so heads chosen here can be
tested on half B with ablation_heads.py:
  framing change    mean change in log-odds(yes) of the demographic prompt
  placebo change    same for its exact-length placebos
  identity-specific framing change minus placebo change, per item, averaged
Heads are ranked by |identity-specific|. Every (head, item, condition) value
is written, so alignment with real rater groups can be computed later on CPU.

Batched: all conditions of BATCH_ITEMS items go through one forward pass,
left-padded, each sequence with its own span mask. Position ids come from the
attention mask so padding does not shift positions.

  MODEL_NAME=... python head_profiling.py --out-dir results/v2/profiling
  Options: --n-items 48 --conditions gender_woman,... --variant prefix
           --layers 0-31 --batch-items 2 --alpha 0.01
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import master_grid as mg
from ablation_heads import set_intervention, DEFAULT_CONDITIONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=mg.ITEMS_CSV)
    ap.add_argument("--n-items", type=int, default=48, help="from split A, stratified by category")
    ap.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    ap.add_argument("--variant", default="prefix")
    ap.add_argument("--layers", default="")
    ap.add_argument("--batch-items", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.01, help="PASTA scaling of non-span attention")
    ap.add_argument("--out-dir", default="results/v2/profiling")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    model_name = mg.MODEL_NAME
    tag = mg.MODEL_TAG
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    matches, _ = mg.build_matching(tok)
    demos = [c for c in a.conditions.split(",") if c]
    conds = list(dict.fromkeys(demos + [p for d in demos for p in matches[d]]))

    items = pd.read_csv(a.items)
    items = items[items.split == "A"]
    share = items.category.value_counts(normalize=True)
    k = (share * a.n_items).round().astype(int)
    k.iloc[0] += a.n_items - k.sum()
    items = pd.concat([g.sample(min(k[c], len(g)), random_state=0) for c, g in items.groupby("category")])
    items = items.sort_values("item_id").reset_index(drop=True)
    print(f"{tag}: {len(items)} split-A items {items.category.value_counts().to_dict()}, "
          f"{len(conds)} conditions {conds}, variant {a.variant}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        attn_implementation="eager").to(device)
    model.eval()
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    lo, hi = (int(x) for x in a.layers.split("-")) if a.layers else (0, n_layers - 1)
    zero_id = tok.encode("0", add_special_tokens=False)[0]
    one_id = tok.encode("1", add_special_tokens=False)[0]
    dhead = mg.DecisionHead(model, zero_id, one_id)

    # --- encode every (item, condition) once ----------------------------------
    batches = []
    for start in range(0, len(items), a.batch_items):
        chunk = items.iloc[start:start + a.batch_items]
        meta, prompts, spans = [], [], []
        for _, item in chunk.iterrows():
            for cond in conds:
                prompt = mg.build_prompt_variant(tok, a.variant, cond, item["text"])
                tokens = mg.tokenize_only(tok, prompt)
                sp = mg.locate_spans(tokens, a.variant, cond)
                why = mg.spans_problem(sp, a.variant, cond)
                if why:
                    raise SystemExit(f"item {item['item_id']} {cond}: {why}")
                meta.append((item["item_id"], cond))
                prompts.append(prompt)
                spans.append(sp[0])
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False,
                  truncation=True, max_length=mg.MAX_LENGTH)
        seq = enc["input_ids"].shape[1]
        mask = torch.zeros((len(prompts), seq), dtype=torch.bool)
        for i, (s, e) in enumerate(spans):
            pad = seq - int(enc["attention_mask"][i].sum())        # left padding shifts every index
            mask[i, pad + s: pad + e + 1] = True
        pos = (enc["attention_mask"].cumsum(-1) - 1).clamp_min(0)
        batches.append((meta, enc["input_ids"].to(device), enc["attention_mask"].to(device),
                        pos.to(device), mask.to(device)))

    @torch.no_grad()
    def logodds(ids, att, pos):
        model(input_ids=ids, attention_mask=att, position_ids=pos)
        return dhead.logodds().cpu().numpy()                # fp32 decision logits

    # --- self-checks ------------------------------------------------------------
    meta, ids, att, pos, mask = batches[0]
    ref = logodds(ids, att, pos)
    single = []
    for i in range(min(3, len(meta))):                     # batched == unbatched (up to bf16 noise)
        n_real = int(att[i].sum())
        single.append(logodds(ids[i:i + 1, -n_real:], att[i:i + 1, -n_real:], pos[i:i + 1, -n_real:])[0])
    print(f"self-check: batched vs single max |d logodds| {np.abs(ref[:len(single)] - single).max():.3e}")
    set_intervention([(l, h) for l in range(n_layers) for h in range(n_heads)], mask, 1.0, "all")
    same = logodds(ids, att, pos)
    set_intervention(None, None, None, "all")
    print(f"self-check: alpha=1 on every head max |d logodds| {np.abs(same - ref).max():.3e}")
    if np.abs(same - ref).max() > 1e-3:
        raise SystemExit("Hook changes the output at alpha=1. Not running.")

    # --- profiling -------------------------------------------------------------
    refs = [logodds(ids, att, pos) for _, ids, att, pos, _ in batches]
    boost = 1.0 / a.alpha
    rows, t0 = [], time.time()
    n_heads_total = (hi - lo + 1) * n_heads
    for li in range(lo, hi + 1):
        for h in range(n_heads):
            for (meta, ids, att, pos, mask), r in zip(batches, refs):
                set_intervention([(li, h)], mask, boost, "all")
                out = logodds(ids, att, pos)
                set_intervention(None, None, None, "all")
                for (item_id, cond), lr, ls in zip(meta, r, out):
                    rows.append((li, h, item_id, cond, lr, ls))
        done = (li - lo + 1) * n_heads
        el = time.time() - t0
        print(f"  layer {li} done  {done}/{n_heads_total} heads  {el / 60:.1f} min  "
              f"ETA {(n_heads_total - done) * el / done / 60:.1f} min", flush=True)

    df = pd.DataFrame(rows, columns=["layer", "head", "item_id", "condition", "logodds_ref", "logodds_steer"])
    df["change"] = df.logodds_steer - df.logodds_ref
    df.insert(0, "model", tag)
    df.to_csv(f"{a.out_dir}/profiling_{tag}.csv.gz", index=False, compression="gzip")

    # per head: identity-specific change = demographic change - mean change of its placebos
    w = df.pivot_table(index=["layer", "head", "item_id"], columns="condition", values="change")
    per = []
    for d in demos:
        spec = w[d] - w[matches[d]].mean(axis=1)
        per.append(pd.DataFrame({"condition": d, "framing_change": w[d],
                                 "placebo_change": w[matches[d]].mean(axis=1), "identity_specific": spec}))
    per = pd.concat(per).reset_index()
    heads = per.groupby(["layer", "head"]).agg(
        identity_specific=("identity_specific", "mean"), identity_specific_sd=("identity_specific", "std"),
        framing_change=("framing_change", "mean"), placebo_change=("placebo_change", "mean"),
        n=("identity_specific", "size")).reset_index()
    heads["identity_specific_t"] = heads.identity_specific / (heads.identity_specific_sd / np.sqrt(heads.n))
    by_demo = per.groupby(["layer", "head", "condition"])["identity_specific"].mean().unstack()
    heads = heads.merge(by_demo.add_prefix("is_").reset_index(), on=["layer", "head"])
    heads = heads.reindex(heads.identity_specific.abs().sort_values(ascending=False).index)
    heads.insert(0, "model", tag)
    heads.to_csv(f"{a.out_dir}/profiling_heads_{tag}.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"\n=== Top 20 heads by |identity-specific change in log-odds| (alpha={a.alpha}) ===")
    print(heads.head(20).round(3).to_string(index=False))
    print(f"\nwrote profiling_{tag}.csv.gz, profiling_heads_{tag}.csv to {a.out_dir}")


if __name__ == "__main__":
    main()
