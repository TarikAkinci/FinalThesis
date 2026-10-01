"""
CPU-only follow-up analysis of attention_probe_segments.csv: statistical
significance for the cross-condition asymmetries flagged as descriptive-only,
attention-vs-prediction-flip correlation, and a per-category breakdown.
No GPU needed -- run on the login node once attention_probe_segments.csv
already exists (from attention_probe_detailed.py).
"""
import pandas as pd
from scipy.stats import wilcoxon

IN_PATH = "results/attention_probing/attention_probe_segments.csv"

# Paired (same 30 items under both conditions) contrasts worth testing --
# these are the asymmetries claimed descriptively in the report.
CONTRAST_PAIRS = [
    ("religion_muslim", "religion_christian"),
    ("region_arab_culture", "region_western_europe"),
    ("gender_man", "gender_woman"),
]

if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)

    print("=== 0. Data quality: rows with an unfound span (NaN pct_item_text or pct_instruction) ===")
    bad = df[df["pct_item_text"].isna() | df["pct_instruction"].isna()]
    if len(bad):
        print(bad[["item_id", "condition"]].to_string(index=False))
    else:
        print("  none -- all spans located cleanly")

    print("\n=== 1. Paired significance: item-text attention vs baseline (per condition) ===")
    baseline_text = df[df.condition == "baseline"][["item_id", "pct_item_text_norm"]].rename(
        columns={"pct_item_text_norm": "baseline_text_norm"})
    for cond in [c for c in df.condition.unique() if c != "baseline"]:
        sub = df[df.condition == cond][["item_id", "pct_item_text_norm"]].merge(baseline_text, on="item_id")
        delta = (sub["pct_item_text_norm"] - sub["baseline_text_norm"]).dropna()
        n_dropped = len(sub) - len(delta)
        stat, p = wilcoxon(delta)
        dropped_note = f"  ({n_dropped} dropped: NaN span)" if n_dropped else ""
        print(f"  {cond:24s} mean_delta={delta.mean():+.4f}  n={len(delta)}  p={p:.4f}{dropped_note}")

    print("\n=== 2. Paired significance: cross-condition demographic-attention asymmetry ===")
    for cond_a, cond_b in CONTRAST_PAIRS:
        a = df[df.condition == cond_a][["item_id", "pct_demographic_norm"]].rename(columns={"pct_demographic_norm": "a"})
        b = df[df.condition == cond_b][["item_id", "pct_demographic_norm"]].rename(columns={"pct_demographic_norm": "b"})
        merged = a.merge(b, on="item_id")
        diff = (merged["a"] - merged["b"]).dropna()
        n_dropped = len(merged) - len(diff)
        stat, p = wilcoxon(diff)
        frac_a_higher = (diff > 0).mean()
        dropped_note = f"  ({n_dropped} dropped: NaN span)" if n_dropped else ""
        print(f"  {cond_a} vs {cond_b}: mean_diff={diff.mean():+.4f}  "
              f"frac_items_{cond_a}_higher={frac_a_higher:.2f}  n={len(diff)}  p={p:.4f}{dropped_note}")

    print("\n=== 3. Attention vs. prediction flips (relative to baseline) ===")
    base_pred = df[df.condition == "baseline"][["item_id", "predicted"]].rename(columns={"predicted": "baseline_pred"})
    merged = df.merge(base_pred, on="item_id")
    merged["flipped"] = merged["predicted"] != merged["baseline_pred"]
    non_baseline = merged[merged.condition != "baseline"]
    print(non_baseline.groupby(["condition", "flipped"])["pct_demographic_norm"].agg(["mean", "count"]))
    print("\nFlip rate by condition:")
    print(non_baseline.groupby("condition")["flipped"].mean().sort_values(ascending=False))

    print("\n=== 4. By-category breakdown ===")
    print(df.groupby(["condition", "category"])[["pct_demographic_norm", "pct_item_text_norm"]].mean())
