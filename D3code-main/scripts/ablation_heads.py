"""
Causal pilot before PASTA: intervene on chosen heads' attention to the
demographic span at the decision token and measure P(yes). GPU.

Intervention: for the chosen (layer, head) pairs, add log(alpha) to the
pre-softmax attention scores from the decision token to the demographic span
tokens. That equals multiplying those heads' unnormalized attention to the span
by alpha and renormalizing, i.e. PASTA-style reweighting restricted to the
decision token. alpha = 0 removes the span from those heads (ablation),
alpha > 1 amplifies it. --scope all applies it to every query position.

Controls, both needed to read the result:
  - random head sets with the same number of heads per layer as the
    candidates (heads drawn from the same layers, candidates excluded)
  - the matched placebo under the same intervention: the identity-specific
    effect is change(demographic) - change(matched placebo) on the same item

Reported per row: P(yes) unframed (baseline), framed without intervention
(full), framed with intervention. The question is whether intervening on the
candidates moves the framed prediction back toward baseline (alpha=0) or
further away (alpha>1) more than random heads do, and more for identity than
for placebo.

Hooks transformers' eager attention by replacing eager_attention_forward in the
Llama and Qwen2 modeling modules (transformers 5.x calls it through a
module-level name). Two self-checks run before the main loop and abort on
failure: alpha=1 on every head must reproduce the unhooked logits, and alpha=0
on one head must drive that head's attention to the span to ~0 while leaving
another head's untouched.

Usage (from D3code-main/scripts):
  MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct python ablation_heads.py \
      --candidates results/v2/analysis/perhead_candidates_Llama-3-1-8B-Instruct.csv \
      --items results/v2/master_grid/master_grid_items_Llama-3-1-8B-Instruct.csv \
      --out-dir results/v2/ablation
  Options: --heads "13:5,14:2" (override candidates), --top-k 10, --n-random 3,
           --alphas 0,2, --variants prefix, --max-items 0, --scope last|all
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import transformers.models.llama.modeling_llama as modeling_llama
import transformers.models.qwen2.modeling_qwen2 as modeling_qwen2
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

import master_grid as mg

STATE = {"heads": {}, "span": None, "bias": 0.0, "scope": "last"}


def _wrap(orig):
    def eager_with_span_bias(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        heads = STATE["heads"].get(module.layer_idx)
        if heads and STATE["span"] is not None:
            q_len, k_len = query.shape[-2], key.shape[-2]
            neg = torch.finfo(query.dtype).min
            if attention_mask is None:
                attention_mask = torch.full((q_len, k_len), neg, dtype=query.dtype, device=query.device) \
                    .triu(1 + k_len - q_len)[None, None]
            elif attention_mask.dtype == torch.bool:
                attention_mask = torch.where(attention_mask, 0.0, neg).to(query.dtype)
            bias = torch.zeros((1, query.shape[1], q_len, k_len), dtype=query.dtype, device=query.device)
            s, e = STATE["span"]
            rows = slice(q_len - 1, q_len) if STATE["scope"] == "last" else slice(None)
            bias[0, heads, rows, s:e + 1] = neg if STATE["bias"] == -math.inf else STATE["bias"]
            attention_mask = attention_mask + bias
        return orig(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
    return eager_with_span_bias


for _mod in (modeling_llama, modeling_qwen2):
    _mod.eager_attention_forward = _wrap(_mod.eager_attention_forward)


def set_intervention(heads, span, alpha, scope):
    """heads: list of (layer, head). alpha=None clears the intervention."""
    if alpha is None or not heads:
        STATE.update(heads={}, span=None)
        return
    by_layer = {}
    for layer, head in heads:
        by_layer.setdefault(int(layer), []).append(int(head))
    STATE.update(heads=by_layer, span=span, scope=scope,
                 bias=-math.inf if alpha == 0 else math.log(alpha))


def parse_heads(s):
    return [tuple(int(v) for v in p.split(":")) for p in s.split(",") if p.strip()]


def random_head_sets(candidates, n_heads, n_sets, seed=0):
    """Same number of heads per layer as `candidates`, drawn from the same
    layers, never a candidate head."""
    rng = np.random.default_rng(seed)
    cand = set(candidates)
    per_layer = pd.Series([l for l, _ in candidates]).value_counts()
    sets = []
    for _ in range(n_sets):
        hs = []
        for layer, k in per_layer.items():
            pool = [h for h in range(n_heads) if (layer, h) not in cand]
            hs += [(int(layer), int(h)) for h in rng.choice(pool, size=min(k, len(pool)), replace=False)]
        sets.append(hs)
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", help="perhead_candidates_<MODEL>.csv")
    ap.add_argument("--heads", help='explicit "layer:head,..." (overrides --candidates)')
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--allow-failed", action="store_true",
                    help="use top-k heads even if none passed the candidate criteria")
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--alphas", default="0,2")
    ap.add_argument("--variants", default="prefix")
    ap.add_argument("--conditions", default="", help="default: matched demographics + their placebos")
    ap.add_argument("--items", required=True, help="master_grid_items_<MODEL>.csv (in-band items)")
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--scope", choices=["last", "all"], default="last")
    ap.add_argument("--out-dir", default="results/ablation")
    a = ap.parse_args()

    model_name = os.environ.get("MODEL_NAME", mg.MODEL_NAME)
    tag = os.environ.get("MODEL_TAG", model_name.rstrip("/").split("/")[-1].replace(".", "-"))
    os.makedirs(a.out_dir, exist_ok=True)

    if a.heads:
        cand = parse_heads(a.heads)
    else:
        c = pd.read_csv(a.candidates)
        if not c["passes"].any() and not a.allow_failed:
            raise SystemExit("No head passed the candidate criteria. That is itself the result; rerun "
                             "with --allow-failed to test the top-k anyway (as an exploratory check).")
        c = c if a.allow_failed else c[c["passes"]]
        cand = list(zip(c["layer"].astype(int), c["head"].astype(int)))[: a.top_k]
    alphas = [float(x) for x in a.alphas.split(",")]
    variants = a.variants.split(",")

    tok = AutoTokenizer.from_pretrained(model_name)
    _, _, demo_to_placebo = mg.build_matched_conditions(tok)
    conds = a.conditions.split(",") if a.conditions else \
        [c for d, p in demo_to_placebo.items() for c in (d, p)]
    conds = list(dict.fromkeys(conds))
    items = pd.read_csv(a.items)
    if a.max_items:
        items = items.head(a.max_items)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        attn_implementation="eager").to(device)
    model.eval()
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    zero_id = tok.encode("0", add_special_tokens=False)[0]
    one_id = tok.encode("1", add_special_tokens=False)[0]
    rand_sets = random_head_sets(cand, n_heads, a.n_random)
    head_sets = {"candidates": cand, **{f"random_{i + 1}": s for i, s in enumerate(rand_sets)}}
    print(f"{tag}: {len(cand)} candidate heads {cand}")
    for name, s in head_sets.items():
        if name != "candidates":
            print(f"  {name}: {s}")

    def encode(variant, cond, text):
        prompt = mg.build_prompt_variant(tok, variant, cond, text)
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=mg.MAX_LENGTH,
                  add_special_tokens=False)["input_ids"].to(device)
        tokens = [tok.decode([t]) for t in ids[0].tolist()]
        demo = None if cond == "baseline" else mg.locate_spans(tokens, variant, cond)[0]
        return ids, demo

    @torch.no_grad()
    def forward(ids, attentions=False):
        out = model(input_ids=ids, output_attentions=attentions)
        logits = out.logits[0, -1].float()
        p = torch.softmax(logits[[zero_id, one_id]], dim=0)[1].item()
        return p, logits, out.attentions if attentions else None

    # --- self-checks ---------------------------------------------------------
    ids, span = encode(variants[0], conds[0], items["text"].iloc[0])
    _, ref_logits, ref_attn = forward(ids, attentions=True)
    set_intervention([(l, h) for l in range(n_layers) for h in range(n_heads)], span, 1.0, a.scope)
    _, chk_logits, _ = forward(ids)
    diff = (chk_logits - ref_logits).abs().max().item()
    l0, h0 = cand[0]
    h_other = (h0 + 1) % n_heads
    set_intervention([(l0, h0)], span, 0.0, a.scope)
    _, _, abl_attn = forward(ids, attentions=True)
    set_intervention(None, None, None, a.scope)
    s, e = span
    on_span = abl_attn[l0][0, h0, -1, s:e + 1].sum().item()
    before = ref_attn[l0][0, h0, -1, s:e + 1].sum().item()
    other_change = (abl_attn[l0][0, h_other, -1] - ref_attn[l0][0, h_other, -1]).abs().max().item()
    print(f"self-check: alpha=1 max logit diff {diff:.2e}; layer {l0} head {h0} span attention "
          f"{before:.4f} -> {on_span:.2e} at alpha=0; head {h_other} max change {other_change:.2e}")
    if diff > 1e-3 or on_span > 1e-3 or other_change > 1e-3:
        raise SystemExit("Self-check failed: the attention hook is not doing what it should for this "
                         "model / transformers version. Not running the main loop.")

    # --- main loop -----------------------------------------------------------
    n_pass = len(items) * (1 + len(variants) * len(conds) * (1 + len(head_sets) * len(alphas)))
    print(f"\n{len(items)} items x {len(variants)} variants x {len(conds)} conditions x "
          f"({len(head_sets)} head sets x {len(alphas)} alphas + 1) = {n_pass} forward passes")
    rows, t0, done = [], time.time(), 0
    for _, item in items.iterrows():
        ids, _ = encode("prefix", "baseline", item["text"])
        p_base, _, _ = forward(ids)
        done += 1
        for variant in variants:
            for cond in conds:
                ids, span = encode(variant, cond, item["text"])
                if span is None:
                    print(f"  span not found: item {item['item_id']} {variant} {cond}, skipped")
                    continue
                p_full, _, _ = forward(ids)
                done += 1
                for set_name, hs in head_sets.items():
                    for alpha in alphas:
                        set_intervention(hs, span, alpha, a.scope)
                        p_int, _, _ = forward(ids)
                        set_intervention(None, None, None, a.scope)
                        done += 1
                        rows.append({"model": tag, "item_id": item["item_id"], "variant": variant,
                                     "condition": cond, "is_placebo": bool(mg.CONDITIONS[cond][3]),
                                     "matched_placebo": demo_to_placebo.get(cond, ""),
                                     "head_set": set_name, "alpha": alpha, "scope": a.scope,
                                     "n_heads": len(hs), "prob_baseline": p_base,
                                     "prob_full": p_full, "prob_int": p_int})
        el = time.time() - t0
        print(f"  {done}/{n_pass} passes  {done / el:.1f}/s  ETA {(n_pass - done) / (done / el) / 60:.1f} min",
              flush=True)

    df = pd.DataFrame(rows)
    df["delta_full"] = df.prob_full - df.prob_baseline
    df["change"] = df.prob_int - df.prob_full
    df.to_csv(f"{a.out_dir}/ablation_{tag}.csv", index=False)

    summ = (df.groupby(["variant", "condition", "is_placebo", "head_set", "alpha"])
            .agg(n=("item_id", "size"), delta_full=("delta_full", "mean"), change=("change", "mean"))
            .reset_index())
    summ["frac_of_framing_shift_undone"] = -summ.change / summ.delta_full

    spec = []
    for (variant, set_name, alpha), x in df.groupby(["variant", "head_set", "alpha"]):
        for demo, placebo in demo_to_placebo.items():
            d = x[x.condition == demo][["item_id", "change"]]
            p = x[x.condition == placebo][["item_id", "change"]]
            m = d.merge(p, on="item_id", suffixes=("_d", "_p"))
            if len(m) < 5:
                continue
            diff = m.change_d - m.change_p
            spec.append({"variant": variant, "head_set": set_name, "alpha": alpha, "condition": demo,
                         "placebo": placebo, "n": len(m), "identity_specific_change": diff.mean(),
                         "p": stats.wilcoxon(diff)[1] if (diff != 0).any() else np.nan})
    spec = pd.DataFrame(spec)
    summ.to_csv(f"{a.out_dir}/ablation_summary_{tag}.csv", index=False)
    spec.to_csv(f"{a.out_dir}/ablation_identity_specific_{tag}.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== Mean change in P(yes) from the intervention (candidates vs random head sets) ===")
    print(summ.pivot_table(index=["variant", "condition", "alpha"], columns="head_set",
                           values="change").round(4).to_string())
    if not spec.empty:
        print("\n=== Identity-specific change: demographic minus its matched placebo ===")
        print(spec.pivot_table(index=["variant", "condition", "alpha"], columns="head_set",
                               values="identity_specific_change").round(4).to_string())
    print(f"\nwrote ablation_{{,summary_,identity_specific_}}{tag}.csv to {a.out_dir}")


if __name__ == "__main__":
    main()
