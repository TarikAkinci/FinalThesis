"""
Sink-excluded, per-layer follow-up to the raw per-layer printout in
attention_probe_detailed.py's log. Raw per-layer demographic shares are
confounded by how much sink that particular layer has (53%-95% across
layers -- see conversation), so this renormalizes the same way the aggregate
analysis already does, then runs a paired Wilcoxon signed-rank test per layer
(religion_muslim vs religion_christian, same 30 items) to check which layers
show a STATISTICALLY, not just descriptively, significant asymmetry.
No GPU needed -- run on the login node once attention_probe_layerwise.csv
already exists (from attention_probe_detailed.py).
"""
import pandas as pd
from scipy.stats import wilcoxon

IN_PATH = "results/attention_probing/attention_probe_layerwise.csv"

if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    df["content_mass"] = 1.0 - df["pct_other_scaffolding"]
    df["pct_demographic_norm"] = df["pct_demographic"] / df["content_mass"]

    print("=== Sink-excluded demographic attention by layer, muslim vs christian ===")
    pivot = df[df.condition.isin(["religion_muslim", "religion_christian"])].groupby(
        ["layer", "condition"])["pct_demographic_norm"].mean().unstack()
    pivot["pct_diff"] = (pivot["religion_muslim"] - pivot["religion_christian"]) / pivot["religion_christian"] * 100
    print(pivot)

    print("\n=== Per-layer paired significance (Wilcoxon, muslim vs christian, n=30 items) ===")
    for l in sorted(df["layer"].unique()):
        a = df[(df.condition == "religion_muslim") & (df.layer == l)][["item_id", "pct_demographic_norm"]].rename(
            columns={"pct_demographic_norm": "a"})
        b = df[(df.condition == "religion_christian") & (df.layer == l)][["item_id", "pct_demographic_norm"]].rename(
            columns={"pct_demographic_norm": "b"})
        merged = a.merge(b, on="item_id")
        diff = (merged["a"] - merged["b"]).dropna()
        if len(diff) < 5 or (diff == 0).all():
            print(f"  layer {l:2d}: insufficient data (n={len(diff)})")
            continue
        stat, p = wilcoxon(diff)
        marker = "  <-- significant (p<0.05)" if p < 0.05 else ""
        print(f"  layer {l:2d}: mean_diff={diff.mean():+.4f}  n={len(diff)}  p={p:.4f}{marker}")
