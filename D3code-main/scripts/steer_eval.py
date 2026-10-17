"""
v3 run C: steering, evaluated on the held-out half (split B). GPU.

Methods
  attention  PASTA-style. In the chosen heads, at every query position,
             attention to the target span is multiplied by alpha and the row
             renormalised (= everything else scaled by 1/alpha). alpha = 100 is
             PASTA's default (0.01 on the rest); alpha < 1 dampens, 0 removes.
  actadd     activation-steering baseline (difference of means). At one layer,
             add coef * d to the hidden state at the answer position, where d
             is the mean difference "framed minus unframed" at that layer and
             position on split-A items, per (variant, condition). coef = 1 adds
             the average framing shift once more, coef = -1 removes it.

Targets (attention method)
  identity   the identity span: arms A (amplify region), B (gender/religion),
             C (distance to the real group, computed afterwards)
  text       the item-text span: arm D. Also applied to the unframed prompt,
             since the question there is over-flagging, not identity.

Controls
  random head sets with the same number of heads per layer (attention method);
  the same intervention on placebo and place-control prompts (their spans are
  steered exactly like the identity spans), so "steering any span" and
  "steering an identity" can be told apart.

Rows written: one per (method, target, head_set, strength, variant, condition,
item) with log-odds of "yes". Strength 1 (attention) / 0 (actadd) is the
unsteered reference. Group rates and distances are added afterwards on CPU
(analysis_steering.py), so the success criterion is applied to raw outputs.

  MODEL_NAME=... python steer_eval.py --heads-file results/v3/profiling/profiling_heads_<MODEL>_prefix.csv
  Options: --top-k 10 --target identity|text --alphas 0,0.5,1,2,4,10,100 --n-random 3
           --method attention|actadd --layer 14 --coefs -1,-0.5,0,0.5,1 --variants prefix,infix,target
           --split B --max-items 0 --batch-items 2 --out-dir results/v3/steering
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import master_grid as mg
from ablation_heads import set_intervention, random_head_sets, parse_heads

DEFAULT_DEMOGRAPHICS = ["region_sub_saharan_africa", "region_arab_culture", "region_western_europe",
                        "region_latin_america", "gender_man", "gender_woman",
                        "religion_christian", "religion_muslim"]


def read_heads(path, top_k):
    """Top-k (layer, head) from a ranked file: the step-1 candidate list
    (column `candidate`) or a profiling ranking (already sorted by effect)."""
    c = pd.read_csv(path)
    if "candidate" in c.columns:
        c = c[c["candidate"].astype(str).str.lower() == "true"]
    return list(zip(c["layer"].astype(int), c["head"].astype(int)))[:top_k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["attention", "actadd"], default="attention")
    ap.add_argument("--target", choices=["identity", "text"], default="identity")
    ap.add_argument("--heads-file")
    ap.add_argument("--heads", help='explicit "layer:head,..." instead of --heads-file')
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--alphas", default="0,0.5,1,2,4,10,100")
    ap.add_argument("--layer", type=int, default=-1, help="actadd layer; default: middle layer")
    ap.add_argument("--coefs", default="-1,-0.5,0,0.5,1")
    ap.add_argument("--n-direction-items", type=int, default=100, help="split-A items for the actadd directions")
    ap.add_argument("--conditions", default=",".join(DEFAULT_DEMOGRAPHICS))
    ap.add_argument("--variants", default="prefix,infix,target")
    ap.add_argument("--items", default=mg.ITEMS_CSV)
    ap.add_argument("--split", default="B")
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--batch-items", type=int, default=2)
    ap.add_argument("--out-dir", default="results/v3/steering")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    tag = mg.MODEL_TAG
    tok = AutoTokenizer.from_pretrained(mg.MODEL_NAME)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    matches, _ = mg.build_matching(tok)
    demos = [c for c in a.conditions.split(",") if c]
    conds = ["baseline"] + list(dict.fromkeys(
        demos + [p for d in demos for p in matches[d]]
        + [p for d in demos for p in mg.PLACE_CONTROL_OF.get(d, [])]))
    variants = [v for v in a.variants.split(",") if v]
    all_items = pd.read_csv(a.items)
    items = all_items if a.split == "all" else all_items[all_items.split == a.split]
    items = items.head(a.max_items) if a.max_items else items

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        mg.MODEL_NAME, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        attn_implementation="eager").to(device)
    model.eval()
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    zero_id = tok.encode("0", add_special_tokens=False)[0]
    one_id = tok.encode("1", add_special_tokens=False)[0]
    dhead = mg.DecisionHead(model, zero_id, one_id)

    def encode(item_rows, variant):
        """Batches of (meta, ids, attention mask, position ids, identity-span
        mask, text-span mask) for every (item, condition) of one variant."""
        batches = []
        for start in range(0, len(item_rows), a.batch_items):
            chunk = item_rows.iloc[start:start + a.batch_items]
            meta, prompts, demo_spans, text_spans = [], [], [], []
            for _, item in chunk.iterrows():
                for cond in conds:
                    v = "prefix" if cond == "baseline" else variant
                    prompt = mg.build_prompt_variant(tok, v, cond, item["text"])
                    sp = mg.locate_spans(mg.tokenize_only(tok, prompt), v, cond)
                    why = mg.spans_problem(sp, v, cond)
                    if why:
                        raise SystemExit(f"item {item['item_id']} {variant} {cond}: {why}")
                    meta.append((item["item_id"], cond))
                    prompts.append(prompt)
                    demo_spans.append(sp[0])
                    text_spans.append(sp[2])
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False,
                      truncation=True, max_length=mg.MAX_LENGTH)
            seq = enc["input_ids"].shape[1]
            dmask = torch.zeros((len(prompts), seq), dtype=torch.bool)
            tmask = torch.zeros((len(prompts), seq), dtype=torch.bool)
            for i in range(len(prompts)):
                pad = seq - int(enc["attention_mask"][i].sum())      # left padding shifts every index
                if demo_spans[i] is not None:
                    dmask[i, pad + demo_spans[i][0]: pad + demo_spans[i][1] + 1] = True
                tmask[i, pad + text_spans[i][0]: pad + text_spans[i][1] + 1] = True
            pos = (enc["attention_mask"].cumsum(-1) - 1).clamp_min(0)
            batches.append((meta, enc["input_ids"].to(device), enc["attention_mask"].to(device),
                            pos.to(device), dmask.to(device), tmask.to(device)))
        return batches

    @torch.no_grad()
    def logodds(ids, att, pos):
        model(input_ids=ids, attention_mask=att, position_ids=pos)
        return dhead.logodds().cpu().numpy()

    rows = []

    def record(meta, out, setting):
        for (item_id, cond), lo in zip(meta, out):
            rows.append((*setting, item_id, cond, float(lo)))

    t0 = time.time()
    if a.method == "attention":
        if a.heads:
            cand = parse_heads(a.heads)
        else:
            cand = read_heads(a.heads_file, a.top_k)
        if not cand:
            raise SystemExit("No heads to steer.")
        head_sets = {"selected": cand, **{f"random_{i + 1}": s for i, s in
                                          enumerate(random_head_sets(cand, n_heads, a.n_random))}}
        alphas = [float(x) for x in a.alphas.split(",")]
        print(f"{tag}: attention steering, target={a.target}, {len(cand)} heads {cand}")
        print(f"  {len(items)} split-{a.split} items x {len(variants)} variants x {len(conds)} conditions x "
              f"({len(head_sets)} head sets x {len([x for x in alphas if x != 1])} alphas + 1 reference)")
        for variant in variants:
            batches = encode(items, variant)
            meta, ids, att, pos, dmask, tmask = batches[0]
            ref0 = logodds(ids, att, pos)
            set_intervention(cand, dmask if a.target == "identity" else tmask, 1.0, "all")
            same = logodds(ids, att, pos)
            set_intervention(None, None, None, "all")
            if np.abs(same - ref0).max() > 1e-3:
                raise SystemExit("Hook changes the output at alpha=1. Not running.")
            for meta, ids, att, pos, dmask, tmask in batches:
                record(meta, logodds(ids, att, pos), ("attention", a.target, "none", 1.0, variant))
            for set_name, hs in head_sets.items():
                for alpha in alphas:
                    if alpha == 1.0:
                        continue
                    for meta, ids, att, pos, dmask, tmask in batches:
                        set_intervention(hs, dmask if a.target == "identity" else tmask, alpha, "all")
                        out = logodds(ids, att, pos)
                        set_intervention(None, None, None, "all")
                        record(meta, out, ("attention", a.target, set_name, alpha, variant))
            print(f"  [{variant}] done  {(time.time() - t0) / 60:.1f} min", flush=True)

    else:   # actadd
        layer = a.layer if a.layer >= 0 else n_layers // 2
        coefs = [float(x) for x in a.coefs.split(",")]
        target_layer = model.model.layers[layer]
        state = {"add": None, "capture": None}

        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            if state["capture"] is not None:
                state["capture"].append(hs[:, -1].detach().float())
            if state["add"] is not None:
                hs = hs.clone()
                hs[:, -1] += state["add"].to(hs.dtype)
                return (hs, *output[1:]) if isinstance(output, tuple) else hs
            return output

        target_layer.register_forward_hook(hook)
        dir_items = all_items[all_items.split == "A"].head(a.n_direction_items)
        print(f"{tag}: activation steering at layer {layer}, coefs {coefs}, "
              f"directions from {len(dir_items)} split-A items")
        for variant in variants:
            # directions: mean over split-A items of h(condition) - h(unframed), per condition
            state["capture"] = []
            metas = []
            for meta, ids, att, pos, _, _ in encode(dir_items, variant):
                logodds(ids, att, pos)
                metas += meta
            H = torch.cat(state["capture"]).cpu().numpy()
            state["capture"] = None
            df_h = pd.DataFrame({"item_id": [m[0] for m in metas], "condition": [m[1] for m in metas]})
            dirs = {}
            bi = df_h.index[df_h.condition == "baseline"].values
            for cond in conds:
                # rows come out item by item in the same order for every condition,
                # so ci and bi are aligned by item
                ci = df_h.index[df_h.condition == cond].values
                dirs[cond] = (H[ci] - H[bi]).mean(0) if cond != "baseline" else np.zeros(H.shape[1])
            batches = encode(items, variant)
            for coef in coefs:
                for meta, ids, att, pos, _, _ in batches:
                    add = torch.tensor(np.stack([coef * dirs[c] for _, c in meta]), device=device)
                    state["add"] = add if coef != 0 else None
                    out = logodds(ids, att, pos)
                    state["add"] = None
                    record(meta, out, ("actadd", f"layer{layer}", "none", coef, variant))
            print(f"  [{variant}] done  {(time.time() - t0) / 60:.1f} min", flush=True)

    df = pd.DataFrame(rows, columns=["method", "target", "head_set", "strength", "variant",
                                     "item_id", "condition", "logodds_yes"])
    df["prob_yes"] = 1 / (1 + np.exp(-df.logodds_yes))
    df.insert(0, "model", tag)
    label = a.label or (f"{a.method}_{a.target}" if a.method == "attention" else "actadd")
    df.insert(1, "set_label", label)
    path = f"{a.out_dir}/steering_{tag}_{label}.csv.gz"
    df.to_csv(path, index=False, compression="gzip")
    print(f"wrote {path} ({len(df)} rows) in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
