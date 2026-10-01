"""
One-off repair for the disagreement_scaled_results.csv produced by the buggy
version of attention_disagreement_scaled.py (merged baseline/gap onto results
using item_id alone, not (item_id, axis) -- 4 items qualified for both
disagreement axes independently, so item_id wasn't a unique key and the merge
cartesian-duplicated rows and could attach the wrong axis's gap value).

The per-row data itself (predicted, prob_yes, pct_* attention columns) was
computed once per (item_id, axis, condition) and is correct -- only the two
merge-derived column sets (baseline_pred/baseline_prob_yes/flipped/delta, and
gap) need to be recomputed, keyed correctly this time. No GPU needed.
"""
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

RESULTS_IN = "results/disagreement_scaled/disagreement_scaled_results.csv"
ITEMS_IN = "results/disagreement_scaled/disagreement_scaled_items.csv"
RESULTS_OUT = "results/disagreement_scaled/disagreement_scaled_results_repaired.csv"

CORE_COLS = ["item_id", "axis", "condition", "prefix_type", "category",
             "predicted", "prob_yes", "pct_demographic", "pct_instruction",
             "pct_item_text", "pct_other_scaffolding", "pct_demographic_norm",
             "pct_instruction_norm", "pct_item_text_norm"]

if __name__ == "__main__":
    raw = pd.read_csv(RESULTS_IN)
    items = pd.read_csv(ITEMS_IN)

    core = raw[CORE_COLS].drop_duplicates(subset=["item_id", "axis", "condition"]).copy()
    print(f"raw rows: {len(raw)}  ->  deduplicated core rows: {len(core)}  (expected 2000)")
    assert len(core) == 2000, "dedup didn't land on the expected row count -- stop and look before trusting this"

    base = core[core.condition == "baseline"][["item_id", "axis", "predicted", "prob_yes"]].rename(
        columns={"predicted": "baseline_pred", "prob_yes": "baseline_prob_yes"})
    results_df = core.merge(base, on=["item_id", "axis"], validate="many_to_one")
    results_df["flipped_vs_baseline"] = results_df["predicted"] != results_df["baseline_pred"]
    results_df["delta_prob_yes"] = results_df["prob_yes"] - results_df["baseline_prob_yes"]
    results_df = results_df.merge(items[["item_id", "axis", "gap"]], on=["item_id", "axis"],
                                   how="left", validate="many_to_one")
    assert len(results_df) == 2000, "post-merge row count drifted -- stop and look"
    results_df.to_csv(RESULTS_OUT, index=False)
    print(f"Saved repaired results to {RESULTS_OUT}")

    AXES = ["gender", "region"]

    print(f"\n=== Flip rate by condition (repaired) ===")
    non_baseline = results_df[results_df.condition != "baseline"]
    print(non_baseline.groupby(["axis", "condition"])["flipped_vs_baseline"].agg(["mean", "sum", "count"]))

    print("\n=== Does the SIZE of the human disagreement gap predict a bigger model shift? (Spearman, repaired) ===")
    for axis in AXES:
        demo_rows = results_df[(results_df.axis == axis) & (results_df.prefix_type == "demographic")]
        per_item = demo_rows.groupby("item_id").agg(
            gap=("gap", "first"),
            mean_abs_delta_prob_yes=("delta_prob_yes", lambda s: s.abs().mean()),
            mean_demo_norm=("pct_demographic_norm", "mean"),
        ).dropna()
        r1, p1 = spearmanr(per_item["gap"], per_item["mean_abs_delta_prob_yes"])
        r2, p2 = spearmanr(per_item["gap"], per_item["mean_demo_norm"])
        print(f"  {axis:8s} (n={len(per_item)}): gap vs |delta P(yes)|: rho={r1:+.3f} p={p1:.4g}   "
              f"gap vs pct_demographic_norm: rho={r2:+.3f} p={p2:.4g}")

    print("\n=== Mean |delta P(yes)| vs baseline, by condition (repaired) ===")
    print(non_baseline.groupby(["axis", "condition"])["delta_prob_yes"].agg(
        ["mean", lambda s: s.abs().mean(), "std"]).rename(columns={"<lambda_0>": "mean_abs_delta"}))

    print("\n=== Placebo check: paired Wilcoxon, real demographic prefix vs placebo prefix, pct_demographic_norm ===")
    for axis in AXES:
        demo_rows = results_df[(results_df.axis == axis) & (results_df.prefix_type == "demographic")]
        demo_mean_per_item = demo_rows.groupby("item_id")["pct_demographic_norm"].mean()
        placebo_rows = results_df[(results_df.axis == axis) & (results_df.prefix_type == "placebo")]
        placebo_per_item = placebo_rows.set_index("item_id")["pct_demographic_norm"]
        merged = pd.DataFrame({"demo": demo_mean_per_item, "placebo": placebo_per_item}).dropna()
        diff = merged["demo"] - merged["placebo"]
        stat, p = wilcoxon(diff)
        print(f"  {axis:8s}: mean_demo={merged['demo'].mean():.4f}  mean_placebo={merged['placebo'].mean():.4f}  "
              f"mean_diff={diff.mean():+.4f}  frac_demo_higher={(diff > 0).mean():.2f}  n={len(diff)}  p={p:.4g}")

    print("\n=== Placebo check: paired Wilcoxon, real demographic vs placebo, |delta P(yes)| ===")
    for axis in AXES:
        demo_rows = results_df[(results_df.axis == axis) & (results_df.prefix_type == "demographic")]
        demo_mean_per_item = demo_rows.groupby("item_id")["delta_prob_yes"].apply(lambda s: s.abs().mean())
        placebo_rows = results_df[(results_df.axis == axis) & (results_df.prefix_type == "placebo")]
        placebo_per_item = placebo_rows.set_index("item_id")["delta_prob_yes"].abs()
        merged = pd.DataFrame({"demo": demo_mean_per_item, "placebo": placebo_per_item}).dropna()
        diff = merged["demo"] - merged["placebo"]
        stat, p = wilcoxon(diff)
        print(f"  {axis:8s}: mean_abs_delta_demo={merged['demo'].mean():.4f}  mean_abs_delta_placebo={merged['placebo'].mean():.4f}  "
              f"mean_diff={diff.mean():+.4f}  n={len(diff)}  p={p:.4g}")
