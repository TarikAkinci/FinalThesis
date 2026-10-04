"""
Length-controlled demographic effects. CPU only, reads master_grid CSVs.

The exact token-matched placebos control length for 4 conditions. This
generalizes the control to every condition, two ways, within each variant:

1. Placebo length line (main estimate). Fit how the shift depends on phrase
   length using only the content-free placebos (item fixed effects, so the
   slope comes from within-item differences). For every condition, the
   length-controlled effect is its shift minus what a placebo of the same
   length would have done on the same item:
       resid_ic = delta_ic - [pbar_i + b_P * (L_c - Lbar_P_i)]
   resid > 0: the phrase pushes toward "offensive" more than length alone.

2. Adjusted means at the average token length (ANCOVA-style). Shift every
   condition's mean to the mean token length of the variant, L_avg:
       adj_c = mean_delta_c - b * (L_c - L_avg)
   reported with b from the placebo line (b_P) and from a pooled fit over all
   conditions with axis fixed effects (b_pool). Two slopes because the placebo
   line rests on few distinct phrases; if the two disagree, length cannot be
   cleanly separated from content in that variant.

Sensitivity: the placebo line is refit leaving each placebo phrase out in turn
(resid_min / resid_max), since one odd placebo (the tea sentence) can move it.
Content-free phrases differ a lot among themselves, so each residual is also
expressed in units of the placebos' own spread around the line
(resid_z_vs_placebo_spread): |z| well above 2 means the condition sits outside
what a content-free phrase of that length does. p_vs_placebo_phrases turns
this into a test with the PHRASE as the unit (prediction interval for one new
placebo phrase, df = n_placebos - 2): the right test for "is this content
special", since resid_p (items as the unit) is significant for almost any
phrase -- the phrase is identical on every item. With only 4-5 placebo
phrases per variant this is weak; the prefix variant gets the full
PLACEBO_POOL (placebo_null block) in newer runs.

The same residualization is applied to sink-excluded demographic attention, and
a per-variant diagnostic reports how much of the condition-to-condition spread
in shift and in attention is explained by length alone.

Length is the in-context span length (demo_span_tokens) when the run recorded
it, otherwise phrase_tokens.

Usage: python analysis_length_control.py [RESULTS_DIR] [OUT_DIR]
"""
import sys

import numpy as np
import pandas as pd

from analysis_utils import load_grid, bh_fdr, cluster_bootstrap_mean, ensure_dir
from scipy import stats

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/master_grid"
OUT_DIR = ensure_dir(sys.argv[2] if len(sys.argv) > 2 else "results/analysis")
OUTCOMES = {"delta": "delta_prob_yes", "attn": "pct_demographic_norm"}


def wide(rows, col):
    """item x condition matrix of `col`, plus condition lengths."""
    w = rows.pivot_table(index="item_id", columns="condition", values=col, aggfunc="first")
    lengths = rows.groupby("condition")["length"].mean()
    return w, lengths


def placebo_slope(w, lengths, placebos):
    """Item fixed-effects slope of outcome on length over the placebo columns."""
    sub = w[placebos].dropna()
    L = lengths[placebos].values
    if len(set(L)) < 2 or sub.empty:
        return np.nan, None
    dY = sub.values - sub.values.mean(axis=1, keepdims=True)
    dL = L - L.mean()
    b = (dY * dL).sum() / (len(sub) * (dL ** 2).sum())
    return float(b), sub


def placebo_residuals(w, lengths, placebos, b):
    pbar = w[placebos].mean(axis=1)
    Lbar = lengths[placebos].mean()
    pred = {c: pbar + b * (lengths[c] - Lbar) for c in w.columns}
    return pd.DataFrame({c: w[c] - pred[c] for c in w.columns})


def pooled_slope(rows, col):
    """Slope on length with item and axis fixed effects (within-item demeaning)."""
    d = rows[["item_id", "axis", "length", col]].dropna()
    if d["length"].nunique() < 2:
        return np.nan
    X = pd.get_dummies(d["axis"], drop_first=True, dtype=float)
    X.insert(0, "length", d["length"].astype(float))
    X = X - X.groupby(d["item_id"]).transform("mean")
    y = d[col] - d.groupby("item_id")[col].transform("mean")
    coef, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
    return float(coef[0])


def distinct_placebos(rows):
    """Placebo conditions with duplicate phrases dropped (e.g. placebo_tok9 is
    the tea sentence, same as placebo_long), so no phrase counts twice."""
    phr = rows[rows.is_placebo].groupby("condition")["phrase"].first().sort_index()
    return sorted(phr[~phr.duplicated()].index)


def analyze_variant(model, variant, rows):
    out, diag = [], {}
    placebos = distinct_placebos(rows)
    all_placebos = set(rows.loc[rows.is_placebo, "condition"])
    axis = rows.groupby("condition")["axis"].first()
    for key, col in OUTCOMES.items():
        if rows[col].isna().all():
            continue
        w, lengths = wide(rows, col)
        b_P, _ = placebo_slope(w, lengths, placebos)
        b_pool = pooled_slope(rows, col)
        resid = placebo_residuals(w, lengths, placebos, b_P) if not np.isnan(b_P) else None

        # leave-one-placebo-out refits
        loo = []
        for drop in placebos:
            keep = [p for p in placebos if p != drop]
            b_k, _ = placebo_slope(w, lengths, keep)
            if not np.isnan(b_k):
                loo.append(placebo_residuals(w, lengths, keep, b_k).mean())
        loo = pd.DataFrame(loo) if loo else None

        cond_lengths = lengths.groupby(level=0).first()
        L_avg = cond_lengths.mean()
        means = w.mean()
        # spread of content-free phrases around their own length line: the scale
        # against which a demographic residual should be judged
        placebo_spread = resid[placebos].mean().std(ddof=1) if resid is not None and len(placebos) >= 3 else np.nan
        diag[f"{key}_placebo_resid_sd"] = placebo_spread
        diag[f"{key}_slope_placebo"] = b_P
        diag[f"{key}_slope_pooled"] = b_pool
        diag[f"{key}_n_placebo_lengths"] = lengths[placebos].nunique()
        if len(cond_lengths) >= 4:
            diag[f"{key}_rho_length_vs_mean"] = stats.spearmanr(cond_lengths, means[cond_lengths.index])[0]
            fit = np.polyfit(cond_lengths, means[cond_lengths.index], 1)
            pred = np.polyval(fit, cond_lengths)
            ss = ((means[cond_lengths.index] - means[cond_lengths.index].mean()) ** 2).sum()
            diag[f"{key}_r2_length_on_means"] = 1 - ((means[cond_lengths.index] - pred) ** 2).sum() / ss if ss > 0 else np.nan

        for c in w.columns:
            rec = {"model": model, "variant": variant, "condition": c, "axis": axis[c],
                   "is_placebo": c in all_placebos, "outcome": key, "length": lengths[c],
                   "length_avg_variant": L_avg, "n_items": int(w[c].notna().sum()),
                   "mean": means[c],
                   "adj_at_avg_length_placebo_slope": means[c] - b_P * (lengths[c] - L_avg) if not np.isnan(b_P) else np.nan,
                   "adj_at_avg_length_pooled_slope": means[c] - b_pool * (lengths[c] - L_avg) if not np.isnan(b_pool) else np.nan}
            if resid is not None:
                r = resid[c].dropna()
                lo, hi = cluster_bootstrap_mean(r.values)
                k = len(placebos)
                Lp = lengths[placebos].values
                sxx = ((Lp - Lp.mean()) ** 2).sum()
                t_phr = np.nan
                if k >= 4 and sxx > 0 and placebo_spread > 0 and c not in all_placebos:
                    t_phr = r.mean() / (placebo_spread * np.sqrt(1 + 1 / k + (lengths[c] - Lp.mean()) ** 2 / sxx))
                rec.update({"resid_mean": r.mean(), "resid_ci_lo": lo, "resid_ci_hi": hi,
                            "resid_z_vs_placebo_spread": r.mean() / placebo_spread if placebo_spread else np.nan,
                            "t_vs_placebo_phrases": t_phr,
                            "p_vs_placebo_phrases": 2 * stats.t.sf(abs(t_phr), df=k - 2) if not np.isnan(t_phr) else np.nan,
                            "resid_items_up": int((r > 0).sum()), "resid_items_down": int((r < 0).sum()),
                            "resid_p": stats.wilcoxon(r)[1] if len(r) >= 5 and (r != 0).any() else np.nan})
            if loo is not None and c in loo.columns:
                rec["resid_min_loo"] = loo[c].min()
                rec["resid_max_loo"] = loo[c].max()
            out.append(rec)
    diag.update({"model": model, "variant": variant, "placebos": "|".join(placebos)})
    return out, diag


def main():
    df = load_grid(RESULTS_DIR)
    grid = df[df.condition != "baseline"]
    rows, diags = [], []
    for (model, variant), g in grid.groupby(["model", "variant"]):
        if g.is_placebo.sum() == 0:
            continue
        r, d = analyze_variant(model, variant, g)
        rows += r
        diags.append(d)
    res = pd.DataFrame(rows)
    res["resid_q"] = np.nan
    for (model, outcome), idx in res.groupby(["model", "outcome"]).groups.items():
        res.loc[idx, "resid_q"] = bh_fdr(res.loc[idx, "resid_p"].values)
    diag = pd.DataFrame(diags)

    # per axis: average length and the length-controlled shift, averaged over conditions
    axes = (res[res.outcome == "delta"]
            .groupby(["model", "variant", "axis"])
            .agg(n_conditions=("condition", "nunique"), mean_length=("length", "mean"),
                 mean_shift=("mean", "mean"), mean_resid=("resid_mean", "mean"),
                 mean_adj_placebo_slope=("adj_at_avg_length_placebo_slope", "mean"),
                 mean_adj_pooled_slope=("adj_at_avg_length_pooled_slope", "mean"))
            .reset_index())

    res.to_csv(f"{OUT_DIR}/length_control_conditions.csv", index=False)
    axes.to_csv(f"{OUT_DIR}/length_control_axes.csv", index=False)
    diag.to_csv(f"{OUT_DIR}/length_control_diagnostics.csv", index=False)
    print(f"wrote {OUT_DIR}/length_control_{{conditions,axes,diagnostics}}.csv")

    pd.set_option("display.width", 220)
    print("\n=== How much does length alone explain? (per variant; condition means) ===")
    cols = [c for c in ["model", "variant", "delta_slope_placebo", "delta_slope_pooled",
                        "delta_r2_length_on_means", "attn_slope_placebo", "attn_r2_length_on_means",
                        "delta_n_placebo_lengths"] if c in diag.columns]
    print(diag[cols].round(4).to_string(index=False))
    print("\n=== Prefix: length-controlled shift per condition (resid vs placebo line) ===")
    p = res[(res.variant == "prefix") & (res.outcome == "delta")]
    cols = ["model", "condition", "length", "mean", "resid_mean", "resid_ci_lo", "resid_ci_hi",
            "resid_z_vs_placebo_spread", "p_vs_placebo_phrases", "resid_min_loo", "resid_max_loo", "resid_q",
            "adj_at_avg_length_pooled_slope"]
    print(p[[c for c in cols if c in p.columns]].sort_values(["model", "resid_mean"], ascending=[True, False])
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
