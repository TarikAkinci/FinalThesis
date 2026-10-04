"""
Sink dynamics by condition. CPU only, reads master_grid CSVs (and the
layerwise .csv.gz when present).

Question: does a real demographic phrase draw attention away from the sink
(chat-template scaffolding, BOS) more or less than a content-free phrase of
the same length? Inserting any phrase adds tokens that can absorb attention, so
the raw drop in sink share is expected; the content question is the paired
difference against the token-matched placebo.

Outputs (OUT_DIR):
  sink_conditions.csv   per (model, variant, condition): mean change in raw sink
                        share vs the item's own baseline, and how the remaining
                        (non-sink) attention is split
  sink_matched.csv      per (model, variant, matched demographic): paired
                        difference to its matched placebo, Wilcoxon p, BH q
  sink_layers.csv       same paired difference per layer (needs layerwise file)

Usage: python analysis_sink_dynamics.py [RESULTS_DIR] [OUT_DIR]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from analysis_utils import load_grid, bh_fdr, cluster_bootstrap_mean, ensure_dir, model_tag_from_path

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/master_grid"
OUT_DIR = ensure_dir(sys.argv[2] if len(sys.argv) > 2 else "results/analysis")


def condition_table(df):
    base = df[df.condition == "baseline"].set_index(["model", "item_id"])["pct_other_scaffolding"]
    g = df[df.condition != "baseline"].copy()
    g["sink_base"] = base.reindex(pd.MultiIndex.from_frame(g[["model", "item_id"]])).values
    g["delta_sink"] = g["pct_other_scaffolding"] - g["sink_base"]
    rows = []
    for (model, variant, cond), x in g.groupby(["model", "variant", "condition"]):
        lo, hi = cluster_bootstrap_mean(x["delta_sink"].values)
        rows.append({"model": model, "variant": variant, "condition": cond,
                     "axis": x["axis"].iloc[0], "is_placebo": bool(x["is_placebo"].iloc[0]),
                     "length": x["length"].mean(), "n": len(x),
                     "sink_raw": x["pct_other_scaffolding"].mean(),
                     "delta_sink_mean": x["delta_sink"].mean(), "delta_sink_ci_lo": lo, "delta_sink_ci_hi": hi,
                     "demo_norm": x["pct_demographic_norm"].mean(),
                     "instr_norm": x["pct_instruction_norm"].mean(),
                     "text_norm": x["pct_item_text_norm"].mean()})
    return pd.DataFrame(rows), g


def matched_table(g):
    rows = []
    for (model, variant), x in g.groupby(["model", "variant"]):
        for demo, d in x[x.matched_placebo != ""].groupby("condition"):
            partner = d["matched_placebo"].iloc[0]
            p = x[x.condition == partner]
            if p.empty:
                continue
            m = d[["item_id", "delta_sink", "length"]].merge(
                p[["item_id", "delta_sink", "length"]], on="item_id", suffixes=("_d", "_p"))
            diff = m["delta_sink_d"] - m["delta_sink_p"]
            lo, hi = cluster_bootstrap_mean(diff.values)
            rows.append({"model": model, "variant": variant, "condition": demo, "matched_placebo": partner,
                         "length_demo": m["length_d"].mean(), "length_placebo": m["length_p"].mean(),
                         "n": len(m), "diff_delta_sink": diff.mean(), "ci_lo": lo, "ci_hi": hi,
                         "p": stats.wilcoxon(diff)[1] if len(diff) >= 5 and (diff != 0).any() else np.nan})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = np.nan
        for model, idx in out.groupby("model").groups.items():
            out.loc[idx, "q"] = bh_fdr(out.loc[idx, "p"].values)
    return out


def layer_table(df):
    """Per-layer sink change vs baseline, demographic minus matched placebo."""
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "master_grid_layerwise_*.csv.gz"))):
        model = model_tag_from_path(path)
        lw = pd.read_csv(path)
        base = lw[lw.condition == "baseline"].set_index(["item_id", "layer"])["pct_other_scaffolding"]
        lw = lw[lw.condition != "baseline"].copy()
        lw["delta_sink"] = lw["pct_other_scaffolding"].values - base.reindex(
            pd.MultiIndex.from_frame(lw[["item_id", "layer"]])).values
        pairs = (df[(df.model == model) & (df.matched_placebo != "")]
                 [["variant", "condition", "matched_placebo"]].drop_duplicates())
        for _, pr in pairs.iterrows():
            d = lw[(lw.variant == pr.variant) & (lw.condition == pr.condition)]
            p = lw[(lw.variant == pr.variant) & (lw.condition == pr.matched_placebo)]
            m = d.merge(p, on=["item_id", "layer"], suffixes=("_d", "_p"))
            for layer, x in m.groupby("layer"):
                diff = x["delta_sink_d"] - x["delta_sink_p"]
                rows.append({"model": model, "variant": pr.variant, "condition": pr.condition,
                             "layer": layer, "diff_delta_sink": diff.mean(),
                             "demo_raw_d": x["pct_demographic_d"].mean(),
                             "demo_raw_p": x["pct_demographic_p"].mean(),
                             "p": stats.wilcoxon(diff)[1] if len(diff) >= 5 and (diff != 0).any() else np.nan})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = np.nan
        for model, idx in out.groupby("model").groups.items():
            out.loc[idx, "q"] = bh_fdr(out.loc[idx, "p"].values)
    return out


def main():
    df = load_grid(RESULTS_DIR)
    cond, g = condition_table(df)
    matched = matched_table(g)
    layers = layer_table(df)
    cond.to_csv(f"{OUT_DIR}/sink_conditions.csv", index=False)
    matched.to_csv(f"{OUT_DIR}/sink_matched.csv", index=False)
    print(f"wrote {OUT_DIR}/sink_conditions.csv, sink_matched.csv")
    if not layers.empty:
        layers.to_csv(f"{OUT_DIR}/sink_layers.csv", index=False)
        print(f"wrote {OUT_DIR}/sink_layers.csv")
    else:
        print("no layerwise files found -- per-layer sink table skipped")

    pd.set_option("display.width", 220)
    print("\n=== Sink change, demographic minus matched placebo (negative = identity pulls MORE from the sink) ===")
    if not matched.empty:
        print(matched.round(4).to_string(index=False))
    print("\n=== Prefix: sink change vs baseline by condition ===")
    p = cond[cond.variant == "prefix"].sort_values(["model", "delta_sink_mean"])
    print(p[["model", "condition", "length", "delta_sink_mean", "delta_sink_ci_lo", "delta_sink_ci_hi",
             "demo_norm", "instr_norm", "text_norm"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
