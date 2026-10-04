"""
Cross-model comparison of the per-head results. CPU only, reads the
perhead_heads_<MODEL>.csv files written by analysis_perhead_candidates.py.

Llama (32 layers x 32 heads) and Qwen (28 x 28) cannot be compared head by
head, so layers are mapped to relative depth (layer / (n_layers - 1)) and
binned. Per bin: share of heads that pass, mean content statistic (placebo-null
|t|), mean behavioural link |rho|. Spearman between the two models' depth
profiles says whether identity-sensitive heads sit at similar relative depth.

Usage: python analysis_perhead_compare.py [ANALYSIS_DIR] [N_BINS]
"""
import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ANALYSIS_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/analysis"
N_BINS = int(sys.argv[2]) if len(sys.argv) > 2 else 8


def profile(heads):
    n_layers = heads.layer.max() + 1
    h = heads.assign(rel_depth=heads.layer / (n_layers - 1))
    h["depth_bin"] = np.minimum((h.rel_depth * N_BINS).astype(int), N_BINS - 1)
    content = "null_t_max_abs" if "null_t_max_abs" in h.columns else "dz_all"
    return h.groupby("depth_bin").agg(
        share_passing=("passes", "mean"),
        content_abs=(content, lambda s: s.abs().mean()),
        behaviour_abs=("rho_all", lambda s: s.abs().mean()),
        n_heads=("head", "size"))


def main():
    paths = sorted(glob.glob(os.path.join(ANALYSIS_DIR, "perhead_heads_*.csv")))
    if not paths:
        raise SystemExit(f"no perhead_heads_*.csv in {ANALYSIS_DIR}; run analysis_perhead_candidates.py first")
    profiles = {}
    for p in paths:
        heads = pd.read_csv(p)
        model = heads.model.iloc[0]
        profiles[model] = profile(heads)
        top = heads[heads.passes] if heads.passes.any() else heads.head(10)
        n_layers = heads.layer.max() + 1
        print(f"{model}: {int(heads.passes.sum())} passing heads; relative depth of "
              f"{'passing' if heads.passes.any() else 'top-10 (none pass)'} heads: "
              f"{np.round(np.sort(top.layer.values / (n_layers - 1)), 2).tolist()}")
    out = pd.concat(profiles, names=["model"]).reset_index()
    out.to_csv(os.path.join(ANALYSIS_DIR, "perhead_depth_profiles.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n=== Depth profiles (bins of relative depth) ===")
    print(out.round(3).to_string(index=False))
    for a, b in itertools.combinations(profiles, 2):
        for col in ("content_abs", "behaviour_abs", "share_passing"):
            both = pd.concat([profiles[a][col], profiles[b][col]], axis=1).dropna()
            x, y = both.iloc[:, 0], both.iloc[:, 1]
            if len(both) >= 3 and x.nunique() > 1 and y.nunique() > 1:
                r, p = stats.spearmanr(x, y)
                print(f"{a} vs {b}  {col:14s} rho={r:.2f}  p={p:.3f}  (n={len(x)} depth bins)")


if __name__ == "__main__":
    main()
