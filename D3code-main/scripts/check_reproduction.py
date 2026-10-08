"""
Do the v2 numbers reproduce the LRZ (v1) numbers where the input is identical?
If not, v1 and v2 results cannot be mixed in the thesis. CPU only.

Two comparisons per model:
  1. baseline P(yes) of every v1 item (the seeded 500): v1 baseline rows vs
     the v2 select_items.py scores, which cover the whole pool;
  2. framed P(yes) where v1 and v2 sent exactly the same prompt: same item,
     same variant, same identity sentence "You are {noun}." (prefix,
     reworded, infix, suffix, embedded, target; the context variants changed
     their elaboration and the religion/identity wording changed, so those are
     not compared).

Identical prompts on different hardware and library versions differ only by
bf16 rounding, so mean |dP| should be well below 0.01 (the verdict threshold). Label flips
are printed too; they can only come from items sitting right at 0.5. The
script prints a verdict but does not gate.

  python check_reproduction.py [--v1 results/master_grid] [--v2 results/v2]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

COMPARABLE_VARIANTS = ["prefix", "reworded", "infix", "suffix", "embedded", "target"]


def compare(a, b, label):
    d = a - b
    flips = int(((a > 0.5) != (b > 0.5)).sum())
    rho = stats.spearmanr(a, b)[0] if len(a) > 2 else np.nan
    ok = len(d) > 0 and d.abs().mean() < 0.01      # flips are informative only: items near 0.5 flip on tiny dP
    print(f"  {label:42s} n={len(d):6d}  mean|dP| {d.abs().mean():.4f}  max|dP| {d.abs().max():.4f}  "
          f"mean dP {d.mean():+.4f}  label flips {flips}  rho {rho:.4f}  -> {'OK' if ok else 'CHECK'}")
    return {"comparison": label, "n": len(d), "mean_abs_dp": d.abs().mean(), "max_abs_dp": d.abs().max(),
            "mean_dp": d.mean(), "label_flips": flips, "spearman": rho, "ok": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default="results/master_grid")
    ap.add_argument("--v2", default="results/v2")
    a = ap.parse_args()
    out = []
    for v1_path in sorted(glob.glob(f"{a.v1}/master_grid_*.csv")):
        tag = os.path.basename(v1_path)[len("master_grid_"):-len(".csv")]
        if tag.startswith(("items_", "layerwise_", "perhead_")):
            continue
        print(f"\n=== {tag} ===")
        v1 = pd.read_csv(v1_path)
        scores_path = f"{a.v2}/scores/scores_{tag}.csv"
        if os.path.exists(scores_path):
            sc = pd.read_csv(scores_path).set_index("item_id")["prob_yes"]
            b1 = v1[v1.condition == "baseline"].set_index("item_id")["prob_yes"]
            common = b1.index.intersection(sc.index)
            r = compare(sc.loc[common], b1.loc[common], "baseline, v1 items (v2 scores)")
            out.append({"model": tag, **r})
        else:
            print(f"  no {scores_path}")

        v2_paths = glob.glob(f"{a.v2}/master_grid/master_grid_{tag}.csv.gz")
        if not v2_paths:
            print(f"  no v2 grid for {tag}")
            continue
        v2 = pd.read_csv(v2_paths[0])
        b2 = v2[v2.condition == "baseline"].set_index("item_id")["prob_yes"]
        b1 = v1[v1.condition == "baseline"].set_index("item_id")["prob_yes"]
        common = b1.index.intersection(b2.index)
        if len(common):
            out.append({"model": tag, **compare(b2.loc[common], b1.loc[common], "baseline, shared items (v2 grid)")})
        v2 = v2[v2.variant.isin(COMPARABLE_VARIANTS) & (v2.condition != "baseline")].copy()
        v2["sentence"] = "You are " + v2["noun"] + "."
        v1c = v1[v1.variant.isin(COMPARABLE_VARIANTS) & (v1.condition != "baseline")]
        m = v2.merge(v1c[["item_id", "variant", "phrase", "prob_yes"]],
                     left_on=["item_id", "variant", "sentence"], right_on=["item_id", "variant", "phrase"],
                     suffixes=("_v2", "_v1"))
        if m.empty:
            print("  no identical framed prompts shared between v1 and v2")
            continue
        out.append({"model": tag, **compare(m.prob_yes_v2, m.prob_yes_v1, "framed, identical prompts")})
        for v, g in m.groupby("variant"):
            out.append({"model": tag, **compare(g.prob_yes_v2, g.prob_yes_v1, f"  framed, {v}")})
    if out:
        path = f"{a.v2}/reproduction_check.csv"
        pd.DataFrame(out).to_csv(path, index=False)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
