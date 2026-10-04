"""
Does the model's response to a demographic framing track what that real rater
group thinks? CPU only.

Real group gap per item (D3CODE raters), as defined in metrics_glossary.md:
    gap_ig = mean(rating_binary | group g) - mean(rating_binary | everyone else)
computed only where the group and the rest each have >= MIN_GROUP_RATERS raters.
D3CODE records rater Region and Gender, not religion, so religion conditions
cannot be checked. identity_arab / identity_european are mapped to the Arab
Culture / Western Europe regions and flagged `approximate`.

Behavioural tests, per (model, variant, condition):
  - across items: Spearman(gap_ig, delta_ic), and the same with the item's mean
    placebo shift (same variant) subtracted, so a general "any phrase moves
    this item" effect does not count as group alignment
Within-item test (prefix only, all 8 regions present there):
  - per item, Spearman across regions between gap_ir and delta_ir: does the
    model order the regions the way the real raters' gaps order them on that
    item? Item-level confounds cancel. Mean rho over items, Wilcoxon vs 0.
    Reported raw and with region main effects removed ("centered"): only the
    centered version shows item-specific tracking; the raw one can come from a
    fixed region ranking shared by model and raters (global_rho_region_means).

Head-level (only with --perhead NPZ): the same within-item, across-region rank
correlation, computed per attention head with the head's per-token attention
to the demographic span in place of delta. Tests whether any head's attention
to the region phrase follows the real raters' gaps. BH-FDR over heads.

Usage:
  python analysis_group_gap.py [RESULTS_DIR] [OUT_DIR]
  python analysis_group_gap.py RESULTS_DIR OUT_DIR --perhead path/to/master_grid_perhead_<MODEL>.npz
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from analysis_utils import (load_grid, load_perhead, bh_fdr, ensure_dir, model_tag_from_path,
                            SEG_DEMO)

D3_RATINGS_CSV = "../dataset/d3-ratings.csv"
D3_RATERS_CSV = "../dataset/d3-raters.csv"
MIN_GROUP_RATERS = 4
MIN_REGIONS_PER_ITEM = 4

CONDITION_TO_GROUP = {
    "region_western_europe": ("Region", "Western Europe", "exact"),
    "region_north_america": ("Region", "North America", "exact"),
    "region_latin_america": ("Region", "Latin America", "exact"),
    "region_arab_culture": ("Region", "Arab Culture", "exact"),
    "region_indian_cultural_sphere": ("Region", "Indian Cultural Sphere", "exact"),
    "region_sinosphere": ("Region", "Sinosphere", "exact"),
    "region_sub_saharan_africa": ("Region", "Sub Saharan Africa", "exact"),
    "region_oceania": ("Region", "Oceania", "exact"),
    "gender_man": ("Gender", "Man", "exact"),
    "gender_woman": ("Gender", "Woman", "exact"),
    "identity_arab": ("Region", "Arab Culture", "approximate"),
    "identity_european": ("Region", "Western Europe", "approximate"),
}


def group_gaps(item_ids):
    """Long table item_id, condition, gap, n_group, n_rest."""
    ratings = pd.read_csv(D3_RATINGS_CSV)
    raters = pd.read_csv(D3_RATERS_CSV)
    r = ratings[ratings.item_id.isin(item_ids)].dropna(subset=["rating_binary"])
    r = r.merge(raters[["rater_id", "Region", "Gender"]], on="rater_id", how="left")
    known = set(raters["Region"].dropna()) | set(raters["Gender"].dropna())
    rows = []
    for cond, (col, val, _) in CONDITION_TO_GROUP.items():
        if val not in known:
            raise ValueError(f"{val!r} is not a value of D3CODE rater column {col}")
        in_g = r[col] == val
        agg = pd.DataFrame({
            "sum_g": r["rating_binary"].where(in_g, 0).groupby(r.item_id).sum(),
            "n_g": in_g.groupby(r.item_id).sum(),
            "sum_all": r.groupby("item_id")["rating_binary"].sum(),
            "n_all": r.groupby("item_id").size(),
        })
        agg["n_rest"] = agg.n_all - agg.n_g
        ok = (agg.n_g >= MIN_GROUP_RATERS) & (agg.n_rest >= MIN_GROUP_RATERS)
        agg = agg[ok]
        gap = agg.sum_g / agg.n_g - (agg.sum_all - agg.sum_g) / agg.n_rest
        rows.append(pd.DataFrame({"item_id": gap.index, "condition": cond, "gap": gap.values,
                                  "n_group": agg.n_g.values, "n_rest": agg.n_rest.values}))
    return pd.concat(rows, ignore_index=True)


def across_items(df, gaps):
    g = df[df.condition != "baseline"]
    placebo_mean = (g[g.is_placebo].groupby(["model", "variant", "item_id"])["delta_prob_yes"]
                    .mean().rename("placebo_delta").reset_index())
    rows = []
    for (model, variant, cond), x in g[g.condition.isin(CONDITION_TO_GROUP)].groupby(["model", "variant", "condition"]):
        m = x.merge(gaps[gaps.condition == cond][["item_id", "gap"]], on="item_id")
        m = m.merge(placebo_mean, on=["model", "variant", "item_id"], how="left")
        rec = {"model": model, "variant": variant, "condition": cond,
               "mapping": CONDITION_TO_GROUP[cond][2], "n_items": len(m)}
        if len(m) >= 8:
            rec["rho_gap_vs_delta"], rec["p_gap_vs_delta"] = stats.spearmanr(m.gap, m.delta_prob_yes)
            corrected = m.delta_prob_yes - m.placebo_delta
            if corrected.notna().sum() >= 8:
                rec["rho_gap_vs_delta_minus_placebo"], rec["p_gap_vs_delta_minus_placebo"] = \
                    stats.spearmanr(m.gap, corrected, nan_policy="omit")
        rows.append(rec)
    out = pd.DataFrame(rows)
    for col in ("p_gap_vs_delta", "p_gap_vs_delta_minus_placebo"):
        if col in out:
            out[col.replace("p_", "q_", 1)] = np.nan
            for model, idx in out.groupby("model").groups.items():
                out.loc[idx, col.replace("p_", "q_", 1)] = bh_fdr(out.loc[idx, col].values)
    return out


def _per_item_rhos(m, value_col):
    out = []
    for _, y in m.groupby("item_id"):
        if len(y) >= MIN_REGIONS_PER_ITEM and y.gap.nunique() > 1 and y[value_col].nunique() > 1:
            out.append(stats.spearmanr(y.gap, y[value_col])[0])
    return np.array(out)


def within_item_regions(df, gaps, value_col="delta_prob_yes"):
    """Per item: Spearman across regions of gap vs value, prefix variant.

    Two versions, because a raw within-item correlation can arise without any
    item-specific tracking: if one region always shifts most AND its raters
    always rate higher, every item shows the same ordering.
      raw       per-item rho on the values as they are
      centered  region main effects removed first (each region's mean delta
                and mean gap over items subtracted), so only item-specific
                deviations can correlate
    `global_rho` is that main-effect alignment itself: Spearman over the 8
    regions between mean delta and mean gap."""
    regions = [c for c, v in CONDITION_TO_GROUP.items() if v[0] == "Region" and v[2] == "exact"]
    rows = []
    for model, x in df[(df.variant == "prefix") & df.condition.isin(regions)].groupby("model"):
        m = x.merge(gaps, on=["item_id", "condition"])
        raw = _per_item_rhos(m, value_col)
        means = m.groupby("condition")[[value_col, "gap"]].mean()
        c = m.copy()
        c[value_col] = c[value_col] - c.condition.map(means[value_col])
        c["gap"] = c["gap"] - c.condition.map(means["gap"])
        centered = _per_item_rhos(c, value_col)
        g_rho, g_p = stats.spearmanr(means[value_col], means.gap)
        rec = {"model": model, "global_rho_region_means": g_rho, "global_p_n8": g_p}
        for name, r in (("raw", raw), ("centered", centered)):
            rec.update({f"{name}_n_items": len(r),
                        f"{name}_mean_rho": r.mean() if len(r) else np.nan,
                        f"{name}_frac_positive": (r > 0).mean() if len(r) else np.nan,
                        f"{name}_p_wilcoxon": stats.wilcoxon(r)[1] if len(r) >= 5 else np.nan})
        rows.append(rec)
    return pd.DataFrame(rows)


def perhead_within_item(npz_path, grid, gaps):
    """Per head: mean over items of the across-region Spearman between real
    gap and the head's per-token attention to the region phrase (prefix),
    both centered by region (item-specific deviations only)."""
    attn, idx = load_perhead(npz_path, grid, segment=SEG_DEMO)
    regions = [c for c, v in CONDITION_TO_GROUP.items() if v[0] == "Region" and v[2] == "exact"]
    sel = (idx.variant == "prefix") & idx.condition.isin(regions)
    idx = idx[sel].reset_index().rename(columns={"index": "row"})
    idx = idx.merge(gaps, on=["item_id", "condition"]).reset_index(drop=True)
    _, n_layers, n_heads = attn.shape
    A_all = attn[idx.row.values].reshape(len(idx), -1) / idx["length"].values[:, None]
    # remove region main effects (see within_item_regions): item-specific deviations only
    for cond in idx.condition.unique():
        sel = (idx.condition == cond).values
        A_all[sel] -= A_all[sel].mean(axis=0)
    idx["gap"] = idx["gap"] - idx.groupby("condition")["gap"].transform("mean")
    rhos = []
    for _, y in idx.groupby("item_id"):
        if len(y) < MIN_REGIONS_PER_ITEM or y.gap.nunique() < 2:
            continue
        A = A_all[y.index.values]
        rg = stats.rankdata(y.gap.values) - (len(y) + 1) / 2
        RA = stats.rankdata(A, axis=0) - (len(y) + 1) / 2
        with np.errstate(invalid="ignore", divide="ignore"):
            rhos.append((rg @ RA) / np.sqrt((rg ** 2).sum() * (RA ** 2).sum(axis=0)))
    R = np.array(rhos)
    mean = np.nanmean(R, axis=0)
    p = np.array([stats.wilcoxon(R[:, k][~np.isnan(R[:, k])])[1]
                  if (~np.isnan(R[:, k])).sum() >= 5 and (R[:, k][~np.isnan(R[:, k])] != 0).any() else np.nan
                  for k in range(R.shape[1])])
    layer, head = np.divmod(np.arange(n_layers * n_heads), n_heads)
    out = pd.DataFrame({"layer": layer, "head": head, "n_items": (~np.isnan(R)).sum(axis=0),
                        "mean_rho_gap_vs_attn": mean, "p": p, "q": bh_fdr(p)})
    return out.sort_values("p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="results/master_grid")
    ap.add_argument("out_dir", nargs="?", default="results/analysis")
    ap.add_argument("--perhead", action="append", default=[], help="per-head npz (repeatable)")
    a = ap.parse_args()
    out_dir = ensure_dir(a.out_dir)

    df = load_grid(a.results_dir)
    gaps = group_gaps(set(df.item_id))
    gaps.to_csv(f"{out_dir}/group_gaps_items.csv", index=False)
    coverage = gaps.groupby("condition").size().rename("items_with_gap")
    print("items with a usable real group gap (>= "
          f"{MIN_GROUP_RATERS} raters in group and rest):\n{coverage.to_string()}")

    across = across_items(df, gaps)
    within = within_item_regions(df, gaps)
    across.to_csv(f"{out_dir}/group_gap_across_items.csv", index=False)
    within.to_csv(f"{out_dir}/group_gap_within_item.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== Across items: does the shift track the group's real gap? ===")
    print(across.round(4).to_string(index=False))
    print("\n=== Within item, across regions (prefix): does the model order regions like the raters do? ===")
    print(within.round(4).to_string(index=False))

    for npz in a.perhead:
        model = model_tag_from_path(npz)
        res = perhead_within_item(npz, df[df.model == model], gaps)
        res.to_csv(f"{out_dir}/group_gap_perhead_{model}.csv", index=False)
        print(f"\n=== {model}: heads whose region attention follows the real gaps (top 10 by p) ===")
        print(res.head(10).round(4).to_string(index=False))
        print(f"heads with q < 0.05: {(res.q < 0.05).sum()} of {len(res)}")


if __name__ == "__main__":
    main()
