"""
Re-analysis of attention_probe_results.csv, fixing two things the raw means
glossed over:
1. Demographic attention is normalized by phrase TOKEN COUNT -- raw summed
   attention mechanically favors longer phrases (region descriptors are
   longer than gender/religion ones), so comparing raw sums across
   conditions with different phrase lengths isn't a fair comparison.
2. Item-text attention uses the PAIRED design we already have (every item
   ran under all 7 conditions) -- per-item delta vs that item's own
   baseline, not just comparing marginal means across conditions.

Tokenizer only, no model weights needed -- runs on the login node in seconds.
"""
from transformers import AutoTokenizer
import pandas as pd

from attention_probe import CONDITIONS, MODEL_NAME

IN_PATH = "attention_probe_results.csv"

if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- 1. normalize demographic attention by phrase token count ---
    phrase_len = {name: len(tok.encode(prefix, add_special_tokens=False)) for name, prefix in CONDITIONS.items() if prefix}
    print("phrase token counts:", phrase_len)

    demo = df.dropna(subset=["pct_attention_on_demographic"]).copy()
    demo["phrase_tokens"] = demo["condition"].map(phrase_len)
    demo["attention_per_token"] = demo["pct_attention_on_demographic"] / demo["phrase_tokens"]

    print("\nRaw mean %% attention on demographic phrase (length-confounded):")
    print(demo.groupby("condition")["pct_attention_on_demographic"].mean().sort_values(ascending=False))

    print("\nAttention PER TOKEN on demographic phrase (length-normalized -- the fair comparison):")
    print(demo.groupby("condition")["attention_per_token"].mean().sort_values(ascending=False))

    # --- 2. paired per-item delta vs that item's own baseline ---
    baseline = df[df["condition"] == "baseline"][["item_id", "pct_attention_on_item_text"]].rename(
        columns={"pct_attention_on_item_text": "baseline_pct"}
    )
    paired = df[df["condition"] != "baseline"].merge(baseline, on="item_id")
    paired["delta_vs_baseline"] = paired["pct_attention_on_item_text"] - paired["baseline_pct"]

    print("\nPaired delta (item text attention minus that item's own baseline), by condition:")
    summary = paired.groupby("condition")["delta_vs_baseline"].agg(["mean", "std", lambda s: (s > 0).mean()])
    summary.columns = ["mean_delta", "std_delta", "frac_items_increased"]
    print(summary.sort_values("mean_delta", ascending=False))
