"""
Analysis of the run-B2 PASTA head profiling (head_profiling.py output). CPU.

Each head was steered on its own toward the identity span (attention to all
other tokens scaled by 0.01) on 48 split-A items, under prefix, infix and
target, for the 4 regions, their length-matched placebos and their
place-name controls. Per head and item:

  identity_specific = mean over regions of [change(region) - mean change(its 2 placebos)]
  place_specific    = mean over regions of [change(region) - mean change(its 2 place controls)]
  framing           = mean over regions of change(region)

"change" is steered minus unsteered log-odds of "yes", both computed in the
same batch, so batch-shape effects cancel; what remains is bf16 rounding
noise. Its size is checked with split-half reliability (24 vs 24 items):
if the head ranking does not replicate across halves, noise dominates.

Per head: mean, t over items, BH-FDR over heads (per model x variant);
reliability across halves; agreement across variants and with the step-1
correlational candidates.

Outputs: results/v3/analysis/profiling_{heads,summary,overlap}.csv
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

from analysis_utils import bh_fdr

PROF = "results/v3/profiling"
OUT = "results/v3/analysis"
STEP1 = "results/v2/analysis/perhead_v3_heads_{model}_raw.csv"
MODELS = ["Llama-3-1-8B-Instruct", "Qwen2-5-7B-Instruct"]
VARIANTS = ["prefix", "infix", "target"]
REGIONS = ["region_sub_saharan_africa", "region_arab_culture", "region_western_europe", "region_latin_america"]
PLACEBOS = {"region_sub_saharan_africa": ["placebo_stairs", "placebo_tea"],
            "region_arab_culture": ["placebo_stairs", "placebo_tea"],
            "region_western_europe": ["placebo_early_riser", "placebo_chess"],
            "region_latin_america": ["placebo_early_riser", "placebo_chess"]}
PLACES = {r: [f"place_researching_{r[7:]}", f"place_mapping_{r[7:]}"] for r in REGIONS}
SEED = 0


def per_item(df):
    """(head x item) frame with the three per-item contrasts."""
    w = df.pivot_table(index=["layer", "head", "item_id"], columns="condition", values="change")
    spec = np.mean([w[r] - w[PLACEBOS[r]].mean(axis=1) for r in REGIONS], axis=0)
    place = np.mean([w[r] - w[PLACES[r]].mean(axis=1) for r in REGIONS], axis=0)
    framing = np.mean([w[r] for r in REGIONS], axis=0)
    plc = np.mean([w[PLACEBOS[r]].mean(axis=1) for r in REGIONS], axis=0)
    return pd.DataFrame({"identity_specific": spec, "place_specific": place, "framing": framing,
                         "placebo": plc}, index=w.index).reset_index()


def head_stats(pi, items):
    x = pi[pi.item_id.isin(items)]
    g = x.groupby(["layer", "head"])
    out = g[["identity_specific", "place_specific", "framing", "placebo"]].mean()
    sd = g[["identity_specific", "place_specific"]].std()
    n = g.size()
    for c in ["identity_specific", "place_specific"]:
        out[f"{c}_t"] = out[c] / (sd[c] / np.sqrt(n))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    heads_all, summ = [], []
    for m in MODELS:
        for v in VARIANTS:
            df = pd.read_csv(f"{PROF}/profiling_{m}_{v}.csv.gz")
            pi = per_item(df)
            items = np.array(sorted(pi.item_id.unique()))
            perm = np.random.default_rng(SEED).permutation(items)
            h1, h2 = perm[: len(items) // 2], perm[len(items) // 2:]
            full, a, b = head_stats(pi, items), head_stats(pi, h1), head_stats(pi, h2)
            full["p"] = 2 * stats.t.sf(full.identity_specific_t.abs(), df=len(items) - 1)
            full["q"] = bh_fdr(full["p"].values)
            full["p_place"] = 2 * stats.t.sf(full.place_specific_t.abs(), df=len(items) - 1)
            full["q_place"] = bh_fdr(full["p_place"].values)
            full["identity_specific_half1"], full["identity_specific_half2"] = a.identity_specific, b.identity_specific
            full["place_specific_half1"], full["place_specific_half2"] = a.place_specific, b.place_specific
            full = full.reset_index()
            full.insert(0, "variant", v)
            full.insert(0, "model", m)
            heads_all.append(full)
            rel_spec = stats.spearmanr(a.identity_specific, b.identity_specific)[0]
            rel_place = stats.spearmanr(a.place_specific, b.place_specific)[0]
            rel_fram = stats.spearmanr(a.framing, b.framing)[0]
            top = full.reindex(full.identity_specific.abs().sort_values(ascending=False).index)
            summ.append({"model": m, "variant": v, "heads": len(full),
                         "split_half_rho_identity_specific": rel_spec,
                         "split_half_rho_place_specific": rel_place, "split_half_rho_framing": rel_fram,
                         "heads_q05_identity": int((full.q < 0.05).sum()),
                         "heads_q05_place": int((full.q_place < 0.05).sum()),
                         "max_abs_identity_specific": full.identity_specific.abs().max(),
                         "top10_mean_abs_identity_specific": top.identity_specific.abs().head(10).mean(),
                         "median_abs_identity_specific": full.identity_specific.abs().median(),
                         "top10_layers": ",".join(map(str, top.layer.head(10).tolist()))})
            print(f"{m} {v}: split-half rho identity {rel_spec:.2f} place {rel_place:.2f} framing {rel_fram:.2f}; "
                  f"q<.05 identity {summ[-1]['heads_q05_identity']} place {summ[-1]['heads_q05_place']}", flush=True)
    heads = pd.concat(heads_all, ignore_index=True)
    heads.to_csv(f"{OUT}/profiling_heads.csv", index=False)
    pd.DataFrame(summ).to_csv(f"{OUT}/profiling_summary.csv", index=False)

    # agreement across variants and with the step-1 correlational candidates
    ov = []
    for m in MODELS:
        h = heads[heads.model == m]
        w = h.pivot_table(index=["layer", "head"], columns="variant", values="identity_specific")
        for i, v1 in enumerate(VARIANTS):
            for v2 in VARIANTS[i + 1:]:
                ov.append({"model": m, "what": f"identity_specific {v1} vs {v2}", "spearman": stats.spearmanr(w[v1], w[v2])[0]})
        s1 = pd.read_csv(STEP1.format(model=m))
        cand = set(zip(s1.loc[s1.candidate, "layer"], s1.loc[s1.candidate, "head"]))
        for v in VARIANTS:
            hv = h[h.variant == v].set_index(["layer", "head"])
            top = set(hv.identity_specific.abs().sort_values(ascending=False).head(len(cand)).index)
            # chance overlap of two random sets of that size among all heads
            ov.append({"model": m, "what": f"top-{len(cand)} profiling ({v}) vs step-1 candidates",
                       "overlap": len(top & cand), "chance": len(cand) ** 2 / len(hv)})
            j = s1.set_index(["layer", "head"])[["t_prefix_family", "rho_prefix_family", "sink_head"]].join(hv[["identity_specific"]])
            j = j[~j.sink_head]
            ov.append({"model": m, "what": f"profiling |identity_specific| ({v}) vs step-1 |content t| (non-sink heads)",
                       "spearman": stats.spearmanr(j.identity_specific.abs(), j.t_prefix_family.abs())[0]})
    pd.DataFrame(ov).to_csv(f"{OUT}/profiling_overlap.csv", index=False)
    print(pd.DataFrame(ov).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
