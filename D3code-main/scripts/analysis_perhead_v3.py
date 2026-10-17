"""
v3 step 1: which attention heads carry REGION content, beyond phrase length,
and does that go with the behavioural effect? CPU only; needs the v2 per-head
npz. Selection data only: split A. Split B is reserved for the causal tests.

Structures (groups):
  prefix_family  prefix, reworded, imagine, consider (pooled: 4 wordings of
                 the same perspective framing, behaviourally near-identical)
  infix          same sentence after the item text
  target         "Do you think this text is offensive for X?"
Regions (v3 decision): Sub-Saharan Africa, Arab Culture, Western Europe,
Latin America. Two 4-token and two 6-token nouns; the strongest region in
both models, the v1 continuity region, the Western reference, and the one
where the two models disagree most.

Per head (attention from the answer position to the identity span, raw and
value-weighted, analysed separately):

  0. sink flag: mean content share (identity + instruction + item text) on
     split-A baselines < SINK_CONTENT; such heads are excluded from candidacy
  1. length diagnostic: Spearman, over the 22 conditions, of mean attention
     mass vs span length (per variant)
  2. content test: per-token attention, condition means over split-A items.
     Fit attention-per-token vs length through the 8 placebos; region
     residual = observed - predicted. Pooled over the 4 regions:
     t = mean residual / (res_sd * sqrt(1/4 + 1/n_p + (Lbar_r - Lbar_p)^2 / Sxx)),
     df = n_p - 2. BH-FDR over heads, per group and kind. (prefix_family:
     condition means are averaged over the 4 templates first.)
  3. behaviour link: per item, dA = region attention - mean of its matched
     placebos, dB = same for delta log-odds, averaged over regions (and
     templates); PARTIAL Spearman over the 250 split-A items, controlling
     item text length and baseline log-odds. Both drive dA and dB (longer
     texts dilute attention to the phrase and weaken the effect: rho -0.41
     with dB, median |rho| 0.35 with dA in Llama), so a plain correlation
     flags almost every head. BH-FDR over heads.
  4. stability: content t and behaviour rho recomputed on two random halves
     of split A (A1, A2).

Candidate rule (frozen before run B; written to the output):
  not a sink head
  AND content q < 0.05 with the same sign in >= 2 of the 3 groups
  AND behaviour q < 0.05 in >= 1 group
  AND prefix_family content t has that sign and |t| >= T_HALF in BOTH halves
  AND prefix_family behaviour rho has the same sign in both halves
Ranked by mean |content t| over groups x max |behaviour rho|.

Usage:  python analysis_perhead_v3.py [--models ...] [--kinds raw vw]
Outputs: results/v2/analysis/perhead_v3_heads_<MODEL>_<KIND>.csv,
         perhead_v3_candidates_<MODEL>_<KIND>.csv, perhead_v3_rule.txt,
         perhead_v3_summary.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from analysis_utils import bh_fdr, spearman_columns

RES = "results/v2/master_grid"
OUT = "results/v2/analysis"
MODELS = ["Llama-3-1-8B-Instruct", "Qwen2-5-7B-Instruct"]
GROUPS = {"prefix_family": ["prefix", "reworded", "imagine", "consider"],
          "infix": ["infix"], "target": ["target"]}
REGIONS = ["region_sub_saharan_africa", "region_arab_culture",
           "region_western_europe", "region_latin_america"]
SINK_CONTENT = 0.05
ALPHA = 0.05
T_HALF = stats.t.ppf(0.975, df=6)       # nominal p < 0.05 at df = n_placebos - 2
SEED = 0

RULE = f"""Candidate rule (v3 step 1, fixed before run B):
- not a sink head (mean content share on split-A baselines >= {SINK_CONTENT})
- content test q < {ALPHA} with the same sign in >= 2 of the 3 groups (prefix_family, infix, target)
- behaviour link q < {ALPHA} in >= 1 group
- prefix_family content t has that sign and |t| >= {T_HALF:.2f} in both random halves of split A
- prefix_family behaviour rho has the same sign in both halves
Regions: {', '.join(REGIONS)}. Data: split A only (250 items). Raw and value-weighted attention analysed separately.
Ranking: mean |content t| over groups x max |behaviour rho| over groups."""


def load(model, kind):
    z = np.load(f"{RES}/master_grid_perhead_{model}.npz", allow_pickle=False)
    att = z["attention_vw" if kind == "vw" else "attention"]
    idx = pd.DataFrame({"item_id": z["item_id"], "variant": z["variant"].astype(str),
                        "condition": z["condition"].astype(str), "length": z["demo_span_tokens"],
                        "text_len": z["text_span_tokens"]})
    grid = pd.read_csv(f"{RES}/master_grid_{model}.csv.gz",
                       usecols=["item_id", "variant", "condition", "split", "is_placebo",
                                "matched_placebos", "delta_logodds_yes", "baseline_logodds_yes"])
    idx = idx.merge(grid, on=["item_id", "variant", "condition"], how="left", validate="1:1")
    idx["is_placebo"] = idx["is_placebo"].astype(str).str.lower() == "true"
    n_layers, n_heads = att.shape[1], att.shape[2]
    base = (idx.condition == "baseline") & (idx.split == "A")
    content = att[base.values][..., :3].astype(np.float32).sum(-1).mean(0).reshape(-1)
    keep = ((idx.split == "A") & idx.variant.isin(sum(GROUPS.values(), []))).values
    demo = att[keep][..., 0].astype(np.float32).reshape(keep.sum(), -1)
    del att
    return demo, idx[keep].reset_index(drop=True), content, n_layers, n_heads


def partial_spearman_columns(x, Y, Z):
    """Spearman between x (n,) and each column of Y (n, k) after removing the
    covariates Z (n, c) from the ranks of both (rank-based partial correlation).
    df = n - 2 - c."""
    rx = stats.rankdata(x)
    RY = stats.rankdata(Y, axis=0)
    RZ = np.column_stack([np.ones(len(x)), stats.rankdata(Z, axis=0)])
    proj = RZ @ np.linalg.pinv(RZ)
    ex = rx - proj @ rx
    EY = RY - proj @ RY
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (ex @ EY) / np.sqrt((ex ** 2).sum() * (EY ** 2).sum(axis=0))
        df = len(x) - 2 - Z.shape[1]
        t = r * np.sqrt(df / np.clip(1 - r ** 2, 1e-12, None))
    return r, 2 * stats.t.sf(np.abs(t), df=df)


def condition_means(A, idx, items, variants):
    """Per-token attention, mean over `items`, averaged over `variants`:
    DataFrame-like (conditions, heads) plus lengths and placebo flags."""
    sel = idx.item_id.isin(items) & idx.variant.isin(variants)
    x = idx[sel]
    per_tok = A[sel.values] / x.length.values[:, None]
    conds = sorted(x.condition.unique())
    means = np.zeros((len(conds), A.shape[1]))
    lengths = np.zeros(len(conds))
    for k, c in enumerate(conds):
        rows = (x.condition == c).values
        means[k] = per_tok[rows].mean(0)
        lengths[k] = x.length.values[rows].mean()
    placebo = np.array([bool(x.loc[x.condition == c, "is_placebo"].iloc[0]) for c in conds])
    return conds, means, lengths, placebo


def content_test(conds, means, lengths, placebo):
    """Pooled-region t per head against the placebo length line; also each
    region's own t. Returns (t_pooled, {region: t})."""
    P = means[placebo]
    Lp = lengths[placebo]
    n = len(Lp)
    X = np.column_stack([np.ones(n), Lp])
    coef, *_ = np.linalg.lstsq(X, P, rcond=None)                       # (2, heads)
    res_sd = np.sqrt(((P - X @ coef) ** 2).sum(0) / (n - 2))
    sxx = ((Lp - Lp.mean()) ** 2).sum()
    resid, per_region = [], {}
    for r in REGIONS:
        k = conds.index(r)
        pr = coef[0] + coef[1] * lengths[k]
        rr = means[k] - pr
        resid.append(rr)
        se1 = res_sd * np.sqrt(1 + 1 / n + (lengths[k] - Lp.mean()) ** 2 / sxx)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_region[r] = rr / se1
    Lr = np.mean([lengths[conds.index(r)] for r in REGIONS])
    se = res_sd * np.sqrt(1 / len(REGIONS) + 1 / n + (Lr - Lp.mean()) ** 2 / sxx)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.mean(resid, axis=0) / se
    return t, per_region


def analyse(model, kind):
    A, idx, content, n_layers, n_heads = load(model, kind)
    items_A = sorted(idx.item_id.unique())
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(items_A)
    halves = {"A1": sorted(perm[: len(perm) // 2]), "A2": sorted(perm[len(perm) // 2:])}
    H = A.shape[1]
    out = pd.DataFrame({"model": model, "kind": kind, "layer": np.arange(H) // n_heads,
                        "head": np.arange(H) % n_heads, "content_share": content,
                        "sink_head": content < SINK_CONTENT})
    out["rel_depth"] = out.layer / (n_layers - 1)

    for v in ["prefix", "infix", "target"]:                       # 1. length diagnostic (mass)
        conds, means, lengths, _ = condition_means(A, idx, items_A, [v])
        mass = means * lengths[:, None]
        rho, _, _ = spearman_columns(lengths, mass)
        out[f"length_rho_{v}"] = rho

    for g, variants in GROUPS.items():                             # 2-4 per structure group
        conds, means, lengths, placebo = condition_means(A, idx, items_A, variants)
        t, per_region = content_test(conds, means, lengths, placebo)
        p = 2 * stats.t.sf(np.abs(t), df=placebo.sum() - 2)
        out[f"t_{g}"], out[f"p_{g}"] = t, p
        out[f"q_{g}"] = bh_fdr(np.where(out.sink_head, np.nan, p))
        for r, tr in per_region.items():
            out[f"t_{g}_{r.replace('region_', '')}"] = tr
        rho, p_rho = partial_spearman_columns(*behaviour_link_args(A, idx, items_A, variants))
        dB_, dA_, _ = behaviour_link_args(A, idx, items_A, variants)
        out[f"rho_raw_{g}"] = spearman_columns(dB_, dA_)[0]
        out[f"rho_{g}"], out[f"p_rho_{g}"] = rho, p_rho
        out[f"q_rho_{g}"] = bh_fdr(np.where(out.sink_head, np.nan, p_rho))
        if g == "prefix_family":
            for h, its in halves.items():
                c2, m2, l2, pl2 = condition_means(A, idx, its, variants)
                out[f"t_{g}_{h}"] = content_test(c2, m2, l2, pl2)[0]
                out[f"rho_{g}_{h}"] = partial_spearman_columns(*behaviour_link_args(A, idx, its, variants))[0]

    tcols = [f"t_{g}" for g in GROUPS]
    sig = np.stack([(out[f"q_{g}"] < ALPHA).values for g in GROUPS], axis=1)
    sgn = np.sign(out[tcols].values)
    pos = (sig & (sgn > 0)).sum(1)
    neg = (sig & (sgn < 0)).sum(1)
    out["content_n_groups"] = np.maximum(pos, neg)
    out["content_sign"] = np.where(pos >= neg, 1, -1)
    out["behaviour_n_groups"] = sum((out[f"q_rho_{g}"] < ALPHA).astype(int) for g in GROUPS)
    s = out.content_sign
    out["stable"] = ((np.sign(out.t_prefix_family_A1) == s) & (np.sign(out.t_prefix_family_A2) == s)
                     & (out.t_prefix_family_A1.abs() >= T_HALF) & (out.t_prefix_family_A2.abs() >= T_HALF)
                     & (np.sign(out.rho_prefix_family_A1) == np.sign(out.rho_prefix_family_A2)))
    out["candidate"] = (~out.sink_head) & (out.content_n_groups >= 2) & (out.behaviour_n_groups >= 1) & out.stable
    out["score"] = out[tcols].abs().mean(axis=1) * out[[f"rho_{g}" for g in GROUPS]].abs().max(axis=1)
    return out.sort_values(["candidate", "score"], ascending=False)


def behaviour_link_args(A, idx, items, variants):
    """Behaviour link inputs over `items`: per item, region-minus-matched-placebo
    attention per head (dA) and region-minus-matched-placebo shift in log-odds
    (dB), averaged over the 4 regions and the group's variants."""
    sel = idx.item_id.isin(items) & idx.variant.isin(variants)
    x = idx[sel].reset_index(drop=True)
    As = A[sel.values]
    pos = {key: i for i, key in enumerate(zip(x.item_id, x.variant, x.condition))}
    items = sorted(items)
    dA = np.zeros((len(items), A.shape[1]))
    dB = np.zeros(len(items))
    n_pairs = 0
    for v in variants:
        for r in REGIONS:
            partners = x.loc[(x.variant == v) & (x.condition == r), "matched_placebos"].iloc[0].split("|")
            ri = [pos[(i, v, r)] for i in items]
            pis = [[pos[(i, v, p)] for i in items] for p in partners]
            dA += As[ri] - np.mean([As[pi] for pi in pis], axis=0)
            dB += x.delta_logodds_yes.values[ri] - np.mean([x.delta_logodds_yes.values[pi] for pi in pis], axis=0)
            n_pairs += 1
    base = x[x.variant == variants[0]].drop_duplicates("item_id").set_index("item_id").loc[items]
    Z = base[["text_len", "baseline_logodds_yes"]].values.astype(float)
    return dB / n_pairs, dA / n_pairs, Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--kinds", nargs="+", default=["raw", "vw"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/perhead_v3_rule.txt", "w") as f:
        f.write(RULE + "\n")
    summ = []
    for m in a.models:
        for k in a.kinds:
            out = analyse(m, k)
            out.to_csv(f"{OUT}/perhead_v3_heads_{m}_{k}.csv", index=False)
            out[out.candidate].to_csv(f"{OUT}/perhead_v3_candidates_{m}_{k}.csv", index=False)
            ok = out[~out.sink_head]
            summ.append({"model": m, "kind": k, "heads": len(out), "sink_heads": int(out.sink_head.sum()),
                         **{f"content_q05_{g}": int((ok[f"q_{g}"] < ALPHA).sum()) for g in GROUPS},
                         **{f"behaviour_q05_{g}": int((ok[f"q_rho_{g}"] < ALPHA).sum()) for g in GROUPS},
                         "content_2plus_groups": int((ok.content_n_groups >= 2).sum()),
                         "stable": int(ok.stable.sum()), "candidates": int(out.candidate.sum()),
                         "median_length_rho_prefix": float(np.nanmedian(out.length_rho_prefix))})
            print(f"{m} {k}: {summ[-1]}", flush=True)
    pd.DataFrame(summ).to_csv(f"{OUT}/perhead_v3_summary.csv", index=False)


if __name__ == "__main__":
    main()
