"""
Which attention heads respond to identity CONTENT, beyond phrase length, and
does that response go with the behavioural shift? Steering-candidate
selection. CPU only; needs the per-head .npz from master_grid.py, so run it
where the npz lives (the GPU instance) and sync back only the CSVs.

Attention value used throughout: the head's attention mass on the demographic
span at the decision token (layer x head, from the npz). Head-averaged mass
mostly tracks how many tokens the phrase has, so every test below removes
length by design rather than by averaging:

  0. Length diagnostic. Across conditions within a variant: how strongly do
     head-averaged mass, head-averaged per-token attention, and each single
     head's mass follow phrase length? Quantifies the problem.

  1. Matched contrast (exact control). For each MATCHED demographic d and its
     token-matched placebo p, per item and head:  dA = A_d - A_p.  Same length,
     same position, same template, so dA != 0 means the head treats identity
     content differently. When the in-context span lengths of d and p differ
     (recorded in the npz), per-token attention is used and the row is flagged.
     Tested per (variant, d), pooled per variant, and pooled over everything
     (per-item mean over (variant, d)): mean, Cohen's dz, Wilcoxon p, BH-FDR
     over heads.

  2. Behavioural link. Same pairs: dB = P(yes)_d - P(yes)_p. Per head, Spearman
     over items of dA vs dB -- does extra attention to identity in this head go
     with a larger identity-specific shift? Pooled fits centre dA and dB within
     each (variant, d) first. BH-FDR over heads.

  3. Placebo null (all demographics, prefix) -- the content test. Per head,
     fit per-token attention against length over the content-free placebo
     phrases (~20 with the placebo_null block), then ask whether a demographic
     phrase falls outside the prediction interval for a new content-free
     phrase of its length: t with df = n_placebos - 2, BH-FDR over
     heads x conditions.

  4. Robustness. Per head, in how many variants the matched contrast is
     significant with the same sign as the pooled one; correlation of head maps
     between variants.

Why the phrase is the unit for content claims: within a condition the
inserted phrase is identical on every item, so a matched contrast over items
mostly tests "these tokens differ from those tokens" and comes out
significant for nearly every head. Whether identity content is special has to
be judged against how much content-free phrases differ among themselves (3).
The matched contrast (1) is kept as an exact-length descriptive measure.

Candidate = placebo-null q < 0.05 for at least one demographic AND
behavioural link q < 0.05, ranked by |t| * |rho|, robustness alongside. If
nothing passes, the top heads by the same score are still written, flagged
passes=False -- that outcome is a result in itself (no head carries identity
content that predicts behaviour).

Usage:
  python analysis_perhead_candidates.py --npz results/.../master_grid_perhead_<MODEL>.npz \
      [--results-dir DIR] [--out-dir DIR] [--top-k 20]
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from analysis_utils import (load_grid, load_perhead, bh_fdr, spearman_columns, paired_columns,
                            ensure_dir, model_tag_from_path, SEG_DEMO)

ALPHA = 0.05


def length_diagnostic(A, idx):
    """Per variant: rank correlation of length with condition-mean attention."""
    rows = []
    for variant, x in idx[idx.condition != "baseline"].groupby("variant"):
        conds = x.groupby("condition")
        if conds.ngroups < 4:
            continue
        L = conds["length"].mean()
        if L.nunique() < 3:
            continue
        mass = np.stack([A[g.index.values].mean(axis=0) for _, g in conds])        # (k, LH)
        per_tok = mass / L.values[:, None]
        rho_heads, _, _ = spearman_columns(L.values, mass)
        rows.append({"variant": variant, "n_conditions": len(L),
                     "rho_length_vs_headavg_mass": stats.spearmanr(L, mass.mean(axis=1))[0],
                     "rho_length_vs_headavg_per_token": stats.spearmanr(L, per_tok.mean(axis=1))[0],
                     "median_rho_length_vs_head_mass": np.nanmedian(rho_heads),
                     "frac_heads_rho_gt_0.7": np.nanmean(rho_heads > 0.7)})
    return pd.DataFrame(rows)


def matched_pairs(A, idx):
    """List of dicts, one per (variant, demographic) with a matched placebo:
    items, dA (n, LH), dB (n,), flags."""
    pairs = []
    for (variant, cond), d in idx[idx.matched_placebo.fillna("") != ""].groupby(["variant", "condition"]):
        partner = d["matched_placebo"].iloc[0]
        p = idx[(idx.variant == variant) & (idx.condition == partner)]
        if p.empty:
            continue
        m = (d.reset_index()[["index", "item_id", "prob_yes", "length"]]
             .merge(p.reset_index()[["index", "item_id", "prob_yes", "length"]], on="item_id", suffixes=("_d", "_p")))
        if len(m) < 3:
            continue
        exact = bool((m.length_d == m.length_p).all())
        Ad, Ap = A[m.index_d.values], A[m.index_p.values]
        if not exact:
            Ad = Ad / m.length_d.values[:, None]
            Ap = Ap / m.length_p.values[:, None]
        pairs.append({"variant": variant, "condition": cond, "placebo": partner,
                      "items": m.item_id.values, "dA": Ad - Ap,
                      "dB": (m.prob_yes_d - m.prob_yes_p).values,
                      "metric": "mass" if exact else "per_token",
                      "length_d": m.length_d.mean(), "length_p": m.length_p.mean()})
    return pairs


def pooled_contrast(pairs):
    """Per-item mean of dA over the given pairs (items present in all)."""
    frames = [pd.DataFrame(pr["dA"], index=pr["items"]) for pr in pairs]
    common = sorted(set.intersection(*(set(f.index) for f in frames)))
    return np.mean([f.loc[common].values for f in frames], axis=0)


def pooled_behaviour(pairs):
    """Stack centred dA / dB over pairs, then per-head Spearman."""
    dA = np.concatenate([pr["dA"] - pr["dA"].mean(axis=0) for pr in pairs])
    dB = np.concatenate([pr["dB"] - pr["dB"].mean() for pr in pairs])
    return spearman_columns(dB, dA)


def placebo_null_z(A, idx, variant="prefix", min_placebos=4):
    """Per head and demographic condition: z of per-token attention against
    the placebos' length line. Returns long DataFrame or None."""
    x = idx[(idx.variant == variant) & (idx.condition != "baseline")]
    if x.empty:
        return None
    phrases = x.groupby("condition")["phrase"].first() if "phrase" in x.columns else None
    conds = x.groupby("condition")
    L = conds["length"].mean()
    per_tok = pd.DataFrame(np.stack([A[g.index.values].mean(axis=0) for _, g in conds]), index=L.index) \
        .div(L, axis=0)
    plc = [c for c in L.index if bool(x.loc[x.condition == c, "is_placebo"].iloc[0])]
    if phrases is not None:
        plc = list(pd.Series(phrases[plc]).drop_duplicates().index)
    if len(plc) < min_placebos or L[plc].nunique() < 2:
        return None
    n = len(plc)
    Lp = L[plc].values
    X = np.column_stack([np.ones(n), Lp])
    coef, *_ = np.linalg.lstsq(X, per_tok.loc[plc].values, rcond=None)
    resid_sd = (per_tok.loc[plc].values - X @ coef).std(axis=0, ddof=2)
    sxx = ((Lp - Lp.mean()) ** 2).sum()
    out = []
    for c in L.index:
        if c in plc or bool(x.loc[x.condition == c, "is_placebo"].iloc[0]):
            continue
        pred = coef[0] + coef[1] * L[c]
        # prediction interval for one new phrase of length L_c
        se = resid_sd * np.sqrt(1 + 1 / n + (L[c] - Lp.mean()) ** 2 / sxx)
        with np.errstate(invalid="ignore", divide="ignore"):
            t = (per_tok.loc[c].values - pred) / se
        out.append(pd.DataFrame({"condition": c, "head_index": np.arange(per_tok.shape[1]),
                                 "t": t, "p": 2 * stats.t.sf(np.abs(t), df=n - 2)}))
    df = pd.concat(out, ignore_index=True)
    df["q"] = bh_fdr(df["p"].values)
    df.attrs["n_placebos"] = n
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--results-dir", default=None, help="defaults to the npz's directory")
    ap.add_argument("--out-dir", default="results/analysis")
    ap.add_argument("--top-k", type=int, default=20)
    a = ap.parse_args()
    out_dir = ensure_dir(a.out_dir)
    model = model_tag_from_path(a.npz)
    grid = load_grid(a.results_dir or os.path.dirname(a.npz), model)
    attn, idx = load_perhead(a.npz, grid, segment=SEG_DEMO)
    n_rows, n_layers, n_heads = attn.shape
    A = attn.reshape(n_rows, -1)
    if "phrase" not in idx.columns:
        idx = idx.merge(grid[["variant", "condition", "phrase"]].drop_duplicates(["variant", "condition"]),
                        on=["variant", "condition"], how="left")
    print(f"{model}: {n_rows} rows, {n_layers} layers x {n_heads} heads")

    # 0. length diagnostic
    diag = length_diagnostic(A, idx)
    diag.insert(0, "model", model)
    diag.to_csv(f"{out_dir}/perhead_length_diagnostic_{model}.csv", index=False)
    print("\n=== 0. Does attention to the phrase just track its length? (across conditions) ===")
    print(diag.round(3).to_string(index=False))

    # 1 + 2. matched contrast and behavioural link
    pairs = matched_pairs(A, idx)
    if not pairs:
        raise SystemExit("No matched demographic/placebo pairs found in this npz.")
    long_rows = []
    for pr in pairs:
        mean, dz, p = paired_columns(pr["dA"])
        rho, p_rho, n = spearman_columns(pr["dB"], pr["dA"])
        long_rows.append(pd.DataFrame({
            "variant": pr["variant"], "condition": pr["condition"], "placebo": pr["placebo"],
            "metric": pr["metric"], "length_demo": pr["length_d"], "length_placebo": pr["length_p"],
            "layer": np.arange(A.shape[1]) // n_heads, "head": np.arange(A.shape[1]) % n_heads,
            "n_items": n, "mean_diff": mean, "dz": dz, "p": p, "q": bh_fdr(p),
            "rho_behaviour": rho, "p_behaviour": p_rho, "q_behaviour": bh_fdr(p_rho)}))
    per_pair = pd.concat(long_rows, ignore_index=True)
    per_pair.to_csv(f"{out_dir}/perhead_matched_pairs_{model}.csv.gz", index=False, compression="gzip")
    inexact = per_pair.loc[per_pair.metric != "mass", ["variant", "condition"]].drop_duplicates()
    if len(inexact):
        print("\nNOTE: span lengths differ in context for these pairs, per-token attention used:\n"
              + inexact.to_string(index=False))

    heads = pd.DataFrame({"layer": np.arange(A.shape[1]) // n_heads, "head": np.arange(A.shape[1]) % n_heads})
    variants = sorted({pr["variant"] for pr in pairs})
    sig_same_sign = np.zeros(A.shape[1], dtype=int)
    dz_maps = {}
    for v in variants:
        vp = [pr for pr in pairs if pr["variant"] == v]
        mean, dz, p = paired_columns(pooled_contrast(vp))
        rho, p_rho, _ = pooled_behaviour(vp)
        heads[f"dz_{v}"], heads[f"q_{v}"] = dz, bh_fdr(p)
        heads[f"rho_{v}"], heads[f"q_rho_{v}"] = rho, bh_fdr(p_rho)
        dz_maps[v] = dz
    mean, dz, p = paired_columns(pooled_contrast(pairs))
    rho, p_rho, n_beh = pooled_behaviour(pairs)
    heads["mean_diff_all"], heads["dz_all"], heads["p_all"], heads["q_all"] = mean, dz, p, bh_fdr(p)
    heads["rho_all"], heads["p_rho_all"], heads["q_rho_all"] = rho, p_rho, bh_fdr(p_rho)
    for v in variants:
        sig_same_sign += ((heads[f"q_{v}"] < ALPHA) & (np.sign(heads[f"dz_{v}"]) == np.sign(heads["dz_all"]))).values
    heads["n_variants_sig_same_sign"] = sig_same_sign
    heads["n_variants"] = len(variants)

    # 3. placebo-null z over all demographics (prefix)
    z = placebo_null_z(A, idx)
    heads = heads.set_index(heads.layer * n_heads + heads["head"])
    if z is not None:
        tw = z.pivot(index="head_index", columns="condition", values="t")
        qw = z.pivot(index="head_index", columns="condition", values="q")
        heads["null_t_max_abs"] = tw.abs().max(axis=1)
        heads["null_condition_at_max"] = tw.abs().idxmax(axis=1)
        heads["null_n_conditions_q05"] = (qw < ALPHA).sum(axis=1)
        heads["null_conditions_q05"] = qw.apply(lambda r: "|".join(r.index[r < ALPHA]), axis=1)
        matched_cols = [c for c in qw.columns if c in {pr["condition"] for pr in pairs}]
        heads["null_n_matched_q05"] = (qw[matched_cols] < ALPHA).sum(axis=1) if matched_cols else 0
        z.assign(layer=z.head_index // n_heads, head=z.head_index % n_heads).drop(columns="head_index") \
            .to_csv(f"{out_dir}/perhead_placebo_null_{model}.csv.gz", index=False, compression="gzip")
        print(f"\nplacebo null: {z.attrs['n_placebos']} distinct content-free phrases (prefix); "
              f"{(z.q < ALPHA).sum()} of {len(z)} head x condition tests at q<{ALPHA}")
        if z.attrs["n_placebos"] < 10:
            print("  WARNING: fewer than 10 placebo phrases -- the null spread is rough. "
                  "Runs with the placebo_null block have ~20.")
        content_ok = heads["null_n_conditions_q05"] > 0
    else:
        print("\nWARNING: no placebo null available (too few placebo phrases under prefix); "
              "falling back to the matched contrast q, which treats items -- not phrases -- as the "
              "unit and is far too lenient for content claims.")
        content_ok = heads.q_all < ALPHA

    # candidates: content beyond what content-free phrases of that length do (phrase as
    # the unit) AND the item-level link between extra identity attention and extra shift
    heads["passes"] = content_ok & (heads.q_rho_all < ALPHA)
    heads["score"] = heads.get("null_t_max_abs", heads.dz_all.abs()) * heads.rho_all.abs()
    heads = heads.sort_values(["passes", "score"], ascending=False)
    heads.insert(0, "model", model)
    heads.to_csv(f"{out_dir}/perhead_heads_{model}.csv", index=False)
    top = heads.head(a.top_k)
    top.to_csv(f"{out_dir}/perhead_candidates_{model}.csv", index=False)

    # 4. robustness across variants
    corr = pd.DataFrame(dz_maps).corr(method="spearman")
    corr.to_csv(f"{out_dir}/perhead_variant_map_corr_{model}.csv")
    layers = heads.groupby("layer")[["dz_all", "rho_all"]].agg(["mean", lambda s: s.abs().max()])
    layers.columns = ["dz_mean", "dz_absmax", "rho_mean", "rho_absmax"]
    layers.to_csv(f"{out_dir}/perhead_layer_summary_{model}.csv")

    pd.set_option("display.width", 220)
    print(f"\n=== Candidate heads ({heads.passes.sum()} of {len(heads)} pass content (placebo null) and "
          f"behaviour at q<{ALPHA}; behaviour n={n_beh} centred pairs) ===")
    cols = ["layer", "head", "null_t_max_abs", "null_condition_at_max", "null_n_conditions_q05",
            "rho_all", "q_rho_all", "dz_all", "n_variants_sig_same_sign", "n_variants", "passes"]
    print(top[[c for c in cols if c in top.columns]].round(4).to_string(index=False))
    print("\n=== Head-map agreement between variants (Spearman of dz over heads) ===")
    print(corr.round(2).to_string())
    print(f"\nwrote perhead_{{heads,candidates,length_diagnostic,matched_pairs,placebo_null,"
          f"variant_map_corr,layer_summary}}_{model}.* to {out_dir}")


if __name__ == "__main__":
    main()
