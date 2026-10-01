"""
Builds the consolidated master table from master_grid.py's output.

CPU only. Reads whichever master_grid_<MODEL>.csv files exist in
results/master_grid/ and emits one row per (model, variant, condition) with
every metric we track, so the whole project's results live in one table
instead of being spread across four studies.

New columns compared to the earlier version of this table:
  - axis / block        which question the row belongs to
  - phrase_tokens       real tokenizer length, for the length confound
  - spearman_entropy    does rater disagreement predict prompt sensitivity?
                        Entropy is now a per-item covariate rather than a
                        selection filter, so this is finally answerable.
  - p_delta_vs_matched  paired test against the EXACT token-matched placebo,
                        not just against the legacy tea sentence.
"""
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

RESULTS_DIR = "results/master_grid"
OUT_CSV = "results/master_table.csv"
OUT_MD = "results/master_table.md"
LEGACY_PLACEBO = "placebo_long"


def binary_entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def paired_p(df, variant, cond_a, cond_b, col):
    """Paired Wilcoxon on the same item_ids, cond_a vs cond_b within a variant."""
    if cond_a == cond_b:
        return np.nan
    a = df[(df.variant == variant) & (df.condition == cond_a)][["item_id", col]]
    b = df[(df.variant == variant) & (df.condition == cond_b)][["item_id", col]]
    m = a.merge(b, on="item_id", suffixes=("_a", "_b")).dropna()
    d = m[f"{col}_a"] - m[f"{col}_b"]
    if len(d) < 5 or (d == 0).all():
        return np.nan
    return wilcoxon(d)[1]


def matched_partner(df, condition):
    """For a demographic condition in the MATCHED block, the placebo_tok{N}
    condition with the same token length. Returns None if there isn't one."""
    row = df[df.condition == condition]
    if row.empty:
        return None
    n_tok = row["phrase_tokens"].iloc[0]
    name = f"placebo_tok{int(n_tok)}"
    return name if (df.condition == name).any() else None


def summarize(df):
    rows = []
    for model in sorted(df.model.unique()):
        m = df[df.model == model]
        base = m[m.condition == "baseline"]
        # baseline rows exist for every item in the seeded sample; the grid runs
        # only on the in-band subset, so restrict the baseline reference row to
        # the items that actually carry conditions
        grid_items = set(m[m.condition != "baseline"].item_id)
        base_grid = base[base.item_id.isin(grid_items)]

        rows.append({
            "model": model, "block": "-", "variant": "-",
            "condition": "baseline (no prefix)", "axis": "baseline",
            "phrase_tokens": 0, "n": len(base_grid),
            "mean_P_yes": base_grid.prob_yes.mean(),
            "pct_pred_offensive": (base_grid.predicted == 1).mean(),
            "mean_margin": (base_grid.prob_yes - 0.5).abs().mean(),
            "mean_pred_entropy": binary_entropy(base_grid.prob_yes.values).mean(),
            "attn_instruction_norm": base_grid.pct_instruction_norm.mean(),
            "attn_item_text_norm": base_grid.pct_item_text_norm.mean(),
            "attn_sink_raw": base_grid.pct_other_scaffolding.mean(),
        })

        grid = m[m.condition != "baseline"]
        for (variant, cond), g in grid.groupby(["variant", "condition"], sort=True):
            d = g.delta_prob_yes
            partner = matched_partner(m, cond)
            rho = np.nan
            if g.entropy.notna().sum() >= 5 and g.entropy.nunique() > 1:
                rho = spearmanr(g.entropy, d)[0]
            rows.append({
                "model": model,
                "block": g.block.iloc[0],
                "variant": variant,
                "condition": cond,
                "axis": g.axis.iloc[0],
                "phrase_tokens": int(g.phrase_tokens.iloc[0]),
                "n": len(g),
                "mean_P_yes": g.prob_yes.mean(),
                "pct_pred_offensive": (g.predicted == 1).mean(),
                "items_up": int((d > 0).sum()),
                "items_down": int((d < 0).sum()),
                "items_flat": int((d == 0).sum()),
                "mean_signed_delta": d.mean(),
                "median_signed_delta": d.median(),
                "sd_signed_delta": d.std(),
                "mean_abs_delta": d.abs().mean(),
                "flip_rate": g.flipped_vs_baseline.mean(),
                "mean_margin": (g.prob_yes - 0.5).abs().mean(),
                "mean_delta_margin": ((g.prob_yes - 0.5).abs()
                                      - (g.baseline_prob_yes - 0.5).abs()).mean(),
                "mean_pred_entropy": binary_entropy(g.prob_yes.values).mean(),
                "mean_delta_entropy": (binary_entropy(g.prob_yes.values)
                                       - binary_entropy(g.baseline_prob_yes.values)).mean(),
                "attn_demographic_norm": g.pct_demographic_norm.mean(),
                "attn_instruction_norm": g.pct_instruction_norm.mean(),
                "attn_item_text_norm": g.pct_item_text_norm.mean(),
                "attn_sink_raw": g.pct_other_scaffolding.mean(),
                "spearman_entropy_vs_delta": rho,
                "p_delta_vs_zero": (wilcoxon(d)[1] if len(d) >= 5 and not (d == 0).all() else np.nan),
                "p_delta_vs_legacy_placebo": paired_p(m, variant, cond, LEGACY_PLACEBO, "delta_prob_yes"),
                "p_attn_vs_legacy_placebo": paired_p(m, variant, cond, LEGACY_PLACEBO, "pct_demographic_norm"),
                "matched_placebo": partner or "",
                "p_delta_vs_matched": (paired_p(m, variant, cond, partner, "delta_prob_yes")
                                       if partner else np.nan),
                "p_attn_vs_matched": (paired_p(m, variant, cond, partner, "pct_demographic_norm")
                                      if partner else np.nan),
            })
    return pd.DataFrame(rows)


def to_markdown(df, path):
    disp = df.copy()
    for c in disp.columns:
        if disp[c].dtype.kind == "f":
            if c.startswith("p_"):
                disp[c] = disp[c].apply(
                    lambda x: "" if pd.isna(x) else (f"{x:.1e}" if x < 1e-3 else f"{x:.3f}"))
            else:
                disp[c] = disp[c].round(4)
    disp = disp.fillna("").astype(str)
    with open(path, "w") as f:
        f.write("| " + " | ".join(disp.columns) + " |\n")
        f.write("| " + " | ".join("---" for _ in disp.columns) + " |\n")
        for _, r in disp.iterrows():
            f.write("| " + " | ".join(r.values) + " |\n")


if __name__ == "__main__":
    paths = sorted(p for p in glob.glob(f"{RESULTS_DIR}/master_grid_*.csv")
                   if "layerwise" not in p and "items" not in p)
    if not paths:
        raise SystemExit(f"No master_grid_*.csv in {RESULTS_DIR}/ -- run master_grid.py first.")
    print("Reading:", *[f"\n  {p}" for p in paths])
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    print(f"{len(df)} rows, models: {sorted(df.model.unique())}")

    table = summarize(df)
    os.makedirs("results", exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    to_markdown(table, OUT_MD)
    print(f"\nwrote {OUT_CSV} and {OUT_MD}  ({len(table)} rows)")

    print("\n=== prefix structure, ranked by mean signed shift ===")
    p = table[(table.variant == "prefix") | (table.condition.str.startswith("baseline"))]
    cols = ["model", "condition", "axis", "phrase_tokens", "n",
            "mean_signed_delta", "items_up", "items_down", "flip_rate",
            "attn_demographic_norm", "p_delta_vs_legacy_placebo"]
    print(p[cols].sort_values("mean_signed_delta", ascending=False).round(4).to_string(index=False))
