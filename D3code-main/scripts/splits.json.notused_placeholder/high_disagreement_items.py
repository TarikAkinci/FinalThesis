"""
Shared item-selection for the disagreement + placebo studies, per your
supervisor's instruction: select the ~100 highest-disagreement items from
WITHIN the seeded 500-item D3CODE evaluation sample (the same one
zeroshot_two_datasets.py's prepare_d3code() reports the headline zero-shot
metrics on), not from the whole D3CODE dataset. Reusing that exact sample
means this disagreement analysis is grounded in the same item pool already
used for the main results, not a separately-drawn one.

Disagreement itself is measured directly from D3CODE's raw per-rater ratings
(Shannon entropy of the empirical rating distribution across the full 0-4
scale, reusing evaluate.py's existing ratings_to_distribution/entropy rather
than reimplementing it -- that module's own docstring says every script
should import from it so numbers stay comparable), not from a specific
demographic group split. That's a deliberate change from the earlier
version, which selected items by a gender-specific or region-specific rater
gap: this version selects items that are controversial in general, then
tests the SAME demographic framings + placebo on all of them, instead of
partitioning items by which axis they were controversial on.

prepare_d3code() returns (text, label) without item_id, so item_id is
recovered by joining back to d3-items.csv on text (unique within the
eval sample, verified: 0 unmatched on the actual data).
"""
import pandas as pd
import numpy as np

from zeroshot_two_datasets import prepare_d3code
from evaluate import ratings_to_distribution, entropy

D3_ITEMS_CSV = "../dataset/d3-items.csv"
D3_RATINGS_CSV = "../dataset/d3-ratings.csv"
MIN_RATERS = 10
N_ITEMS = 100


FILTERED_OUT_PATH = "results/high_disagreement_items_filtered.csv"


def rank_by_entropy(min_raters=MIN_RATERS):
    """Returns EVERY qualifying item from the seeded 500-item D3CODE eval
    sample (not capped at N_ITEMS), sorted by entropy descending. Pure
    pandas, no GPU needed -- the entropy-only ranking. See
    apply_baseline_probability_filter() for the GPU-dependent second stage
    that also requires baseline P(yes) to be in a reasonable band."""
    eval_sample, _ = prepare_d3code()
    items = pd.read_csv(D3_ITEMS_CSV)
    items_unique_text = items.drop_duplicates(subset="text", keep=False)
    merged = eval_sample.merge(items_unique_text[["item_id", "text", "category"]], on="text", how="left")
    if merged["item_id"].isna().any():
        raise ValueError(f"{merged['item_id'].isna().sum()} eval-sample items didn't match a unique item_id by text")

    ratings = pd.read_csv(D3_RATINGS_CSV)
    rows = []
    for _, row in merged.iterrows():
        item_ratings = ratings[ratings.item_id == row["item_id"]]
        raw = [r for r in item_ratings["rating_raw"].dropna().tolist() if r in (0, 1, 2, 3, 4)]
        if len(raw) < min_raters:
            continue
        dist = ratings_to_distribution(raw)
        rows.append({
            "item_id": row["item_id"], "text": row["text"], "category": row["category"],
            "n_raters": len(raw), "entropy": entropy(dist), "variance": float(np.var(raw)),
            "p_offensive": item_ratings["rating_binary"].dropna().mean(),
        })

    df = pd.DataFrame(rows)
    return df.sort_values("entropy", ascending=False).reset_index(drop=True)


def select_high_disagreement_items(n_items=N_ITEMS, min_raters=MIN_RATERS):
    """Entropy-only selection (no baseline-probability filter) -- the
    n_items highest-entropy items from the seeded 500-item eval sample.
    Kept for backward compatibility / standalone inspection; the actual
    experiment scripts use load_filtered_items() instead, which also
    requires baseline P(yes) to be in a reasonable band (see
    select_items_with_baseline_filter.py)."""
    return rank_by_entropy(min_raters).head(n_items).reset_index(drop=True)


def load_filtered_items(path=FILTERED_OUT_PATH):
    """Loads the item list produced by select_items_with_baseline_filter.py
    -- entropy-ranked AND baseline-probability-band-filtered. Run that
    script (needs a GPU) before calling this."""
    return pd.read_csv(path)


if __name__ == "__main__":
    df = select_high_disagreement_items()
    print(f"Selected {len(df)} high-disagreement items from the seeded 500-item eval sample")
    print(f"entropy range: {df.entropy.min():.3f} - {df.entropy.max():.3f}")
    print(df["category"].value_counts())
