"""
One frozen item list for every model and every experiment (v2 design).

Why: the LRZ grid filtered items per model on that model's own baseline
P(yes), so Llama kept 236 items and Qwen 64 (only 54 shared). Qwen puts most
items above 0.98, so its in-band set was smaller and different. Comparisons
across models then ran on different items with different n.

Here the pool is the whole D3CODE dataset (unique texts, >= MIN_RATERS
ratings), not the seeded 500-item eval sample. Two steps:

  score   GPU, once per model. Baseline P(yes) and log-odds for every pool
          item, same prompt, same eager attention path as master_grid.py
          (output_attentions off; master_grid.py's noise check reports how
          much that changes the numbers).
            MODEL_NAME=... python select_items.py score --out results/v2/scores

  select  CPU, after both models are scored. Keeps items whose baseline
          P(yes) lies in [0.02, 0.98] for EVERY scored model, draws N_TARGET
          of them (seeded) with category quotas proportional to the full
          pool, and assigns a 50/50 selection/test split stratified by
          category. Writes the item list that every later script reads.
            python select_items.py select --scores results/v2/scores --out items/items_v2.csv

The band filter stays (items pinned near 0 or 1 have no headroom, a real,
diagnosed confound). Requiring it for both models is what makes the set
shared; the cost is a bias toward items Qwen is unsure about (milder items,
fewer social-group ones). That bias is printed by `select` and belongs in the
limitations.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from evaluate import ratings_to_distribution, entropy

D3_ITEMS_CSV = "../dataset/d3-items.csv"
D3_RATINGS_CSV = "../dataset/d3-ratings.csv"
MIN_RATERS = 10
MIN_BASELINE_PROB = 0.02
MAX_BASELINE_PROB = 0.98
N_TARGET = int(os.environ.get("N_TARGET", "400"))
SEED = 42


def load_item_pool(min_raters=MIN_RATERS):
    """Every D3CODE item with a unique text and >= min_raters valid ratings,
    with the per-item rating statistics the analyses use as covariates."""
    items = pd.read_csv(D3_ITEMS_CSV)
    items = items.drop_duplicates(subset="text", keep=False)   # duplicated texts are ambiguous
    ratings = pd.read_csv(D3_RATINGS_CSV)
    ratings = ratings[ratings.item_id.isin(set(items.item_id))]
    rows = []
    for item_id, r in ratings.groupby("item_id"):
        raw = [x for x in r["rating_raw"].dropna().tolist() if x in (0, 1, 2, 3, 4)]
        if len(raw) < min_raters:
            continue
        rows.append({"item_id": item_id, "n_raters": len(raw),
                     "entropy": entropy(ratings_to_distribution(raw)),
                     "variance": float(np.var(raw)),
                     "p_offensive": r["rating_binary"].dropna().mean()})
    stats_df = pd.DataFrame(rows)
    pool = items.rename(columns={"sub-category": "sub_category"}).merge(stats_df, on="item_id")
    return pool[["item_id", "text", "category", "sub_category", "n_raters",
                 "entropy", "variance", "p_offensive"]].sort_values("item_id").reset_index(drop=True)


def score(out_dir):
    import torch
    import master_grid as mg
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(out_dir, exist_ok=True)
    pool = load_item_pool()
    print(f"{len(pool)} pool items (unique text, >= {MIN_RATERS} raters)")
    tok = AutoTokenizer.from_pretrained(mg.MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        mg.MODEL_NAME, dtype=torch.bfloat16, attn_implementation="eager").to(device)
    model.eval()
    zero_id = tok.encode("0", add_special_tokens=False)[0]
    one_id = tok.encode("1", add_special_tokens=False)[0]
    dhead = mg.DecisionHead(model, zero_id, one_id)
    rows = []
    with torch.no_grad():
        for i, item in enumerate(pool.itertuples()):
            prompt = mg.build_prompt_variant(tok, "prefix", "baseline", item.text)
            ids = tok(prompt, return_tensors="pt", truncation=True, max_length=mg.MAX_LENGTH,
                      add_special_tokens=False).to(device)
            model(**ids)
            lo = dhead.logodds()[0].item()                  # fp32 decision logits
            rows.append({"item_id": item.item_id, "prob_yes": 1 / (1 + np.exp(-lo)), "logodds_yes": lo})
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(pool)}", flush=True)
    out = pool.merge(pd.DataFrame(rows), on="item_id")
    out.insert(0, "model", mg.MODEL_TAG)
    path = f"{out_dir}/scores_{mg.MODEL_TAG}.csv"
    out.to_csv(path, index=False)
    inb = out.prob_yes.between(MIN_BASELINE_PROB, MAX_BASELINE_PROB)
    print(f"wrote {path}: {inb.sum()} of {len(out)} items in band for {mg.MODEL_TAG}")


def _quotas(available, shares, n):
    """Largest-remainder allocation of n over categories by `shares`, capped
    by what is available; any shortfall is redistributed to categories that
    still have items."""
    quota = pd.Series(0, index=shares.index)
    remaining = n
    active = [c for c in shares.index if available.get(c, 0) > 0]
    while remaining > 0 and active:
        s = shares[active] / shares[active].sum()
        raw = s * remaining
        add = np.floor(raw).astype(int)
        rest = remaining - add.sum()
        for c in (raw - add).sort_values(ascending=False).index[:rest]:
            add[c] += 1
        for c in active:
            quota[c] += min(add[c], available[c] - quota[c])
        remaining = n - quota.sum()
        active = [c for c in active if quota[c] < available[c]]
    return quota


def select(scores_dir, out_path, n_target=N_TARGET):
    paths = sorted(glob.glob(f"{scores_dir}/scores_*.csv"))
    if len(paths) < 2:
        raise SystemExit(f"Need the scores of both models in {scores_dir}, found {paths}")
    scores = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    models = sorted(scores.model.unique())
    wide = scores.pivot(index="item_id", columns="model", values="prob_yes")
    in_band = wide.apply(lambda c: c.between(MIN_BASELINE_PROB, MAX_BASELINE_PROB)).all(axis=1)
    pool = scores.drop_duplicates("item_id").drop(columns=["model", "prob_yes", "logodds_yes"]) \
        .set_index("item_id")
    eligible = pool.loc[in_band[in_band].index]
    print(f"models: {models}")
    for m in models:
        print(f"  in band for {m}: {wide[m].between(MIN_BASELINE_PROB, MAX_BASELINE_PROB).sum()}")
    print(f"  in band for all: {len(eligible)} of {len(pool)}")

    shares = pool.category.value_counts(normalize=True)
    available = eligible.category.value_counts()
    n = min(n_target, len(eligible))
    n -= n % 2                                   # even, so the split is exactly 50/50
    if n < n_target:
        print(f"WARNING: only {len(eligible)} eligible items, using N={n} instead of {n_target}")
    quota = _quotas(available.to_dict(), shares, n)
    rng = np.random.default_rng(SEED)
    picked = []
    for cat, k in quota.items():
        ids = sorted(eligible.index[eligible.category == cat])
        picked += list(rng.choice(ids, size=int(k), replace=False))
    sel = eligible.loc[sorted(picked)].reset_index()

    # 50/50 split, stratified by category; within a category items are shuffled
    # and alternated so A and B differ by at most one item per category, and the
    # odd ones are balanced so the halves end up exactly equal
    sel["split"] = ""
    leftovers = []
    for cat, g in sel.groupby("category"):
        order = rng.permutation(g.index.values)
        half = len(order) // 2
        sel.loc[order[:half], "split"] = "A"
        sel.loc[order[half:2 * half], "split"] = "B"
        leftovers += list(order[2 * half:])
    for i, idx in enumerate(leftovers):
        sel.loc[idx, "split"] = "A" if i % 2 == 0 else "B"

    for m in models:
        s = scores[scores.model == m].set_index("item_id")
        sel[f"baseline_prob_{m}"] = s.loc[sel.item_id, "prob_yes"].values
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sel.to_csv(out_path, index=False)

    print(f"\nwrote {out_path}: N={len(sel)}  split {sel.split.value_counts().to_dict()}")
    summary = pd.DataFrame({"pool_share": shares, "eligible": available,
                            "selected": sel.category.value_counts(),
                            "A": sel[sel.split == "A"].category.value_counts(),
                            "B": sel[sel.split == "B"].category.value_counts()}).fillna(0)
    print(summary.round(3).to_string())
    print(f"\nselection bias check (pool vs selected):")
    print(f"  mean human offensive fraction  {pool.p_offensive.mean():.3f} vs {sel.p_offensive.mean():.3f}")
    print(f"  mean rating entropy            {pool.entropy.mean():.3f} vs {sel.entropy.mean():.3f}")
    for m in models:
        print(f"  mean baseline P(yes) {m}: {sel[f'baseline_prob_{m}'].mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("score")
    s1.add_argument("--out", default="results/v2/scores")
    s2 = sub.add_parser("select")
    s2.add_argument("--scores", default="results/v2/scores")
    s2.add_argument("--out", default="items/items_v2.csv")
    s2.add_argument("--n", type=int, default=N_TARGET)
    a = ap.parse_args()
    if a.cmd == "score":
        score(a.out)
    else:
        select(a.scores, a.out, a.n)
