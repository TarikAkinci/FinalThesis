"""
Freezes the heads for run C (steering on split B), from the run-B2 profiling
on split A. CPU only. Written before any split-B steering result exists.

Rule
  pool:      heads whose region effect under single-head PASTA steering is
             beyond the 99th percentile of BOTH nulls in at least one format
             (prefix, infix, target):
               - placebo-pair null: |change(placebo) - change(other placebo of the same length)|
               - place-pair null:   |change(researching X) - change(mapping X)|
             i.e. the head separates "someone from X" both from content-free
             phrases and from the bare place name X more than two
             interchangeable phrases are separated.
  dampeners: place_specific < 0 in BOTH prefix and infix (turning the head up
             toward the identity lowers P(offensive) for the region more than
             for its place controls)
  amplifiers: place_specific > 0 in both prefix and infix
  ranking:   mean |place_specific| over prefix and infix
The two sets are steered separately: mixed, their effects would cancel.

Outputs: results/v3/analysis/heads_<MODEL>_{dampeners,amplifiers}.csv, heads_rule.txt
"""
import pandas as pd

OUT = "results/v3/analysis"
MODELS = ["Llama-3-1-8B-Instruct", "Qwen2-5-7B-Instruct"]
VARIANTS = ["prefix", "infix", "target"]

RULE = __doc__.split("Rule", 1)[1].split("Outputs:")[0]


def main():
    h = pd.read_csv(f"{OUT}/profiling_heads.csv")
    with open(f"{OUT}/heads_rule.txt", "w") as f:
        f.write("Run-C head selection (split A profiling, frozen before split B)\n" + RULE)
    for m in MODELS:
        pool = set()
        for v in VARIANTS:
            n = pd.read_csv(f"{OUT}/profiling_null_{m}_{v}.csv")
            pool |= set(zip(n.loc[n.beyond_placebo_null & n.beyond_place_null, "layer"],
                            n.loc[n.beyond_placebo_null & n.beyond_place_null, "head"]))
        w = h[h.model == m].pivot_table(index=["layer", "head"], columns="variant", values="place_specific")
        w = w.loc[sorted(pool)]
        w["rank_score"] = w[["prefix", "infix"]].abs().mean(axis=1)
        for name, mask in [("dampeners", (w.prefix < 0) & (w.infix < 0)),
                           ("amplifiers", (w.prefix > 0) & (w.infix > 0))]:
            s = w[mask].sort_values("rank_score", ascending=False).reset_index()
            s.insert(0, "model", m)
            s.to_csv(f"{OUT}/heads_{m}_{name}.csv", index=False)
            print(f"{m} {name}: {list(zip(s.layer, s['head']))}")
        rest = w[~(((w.prefix < 0) & (w.infix < 0)) | ((w.prefix > 0) & (w.infix > 0)))]
        print(f"{m} pool {len(w)}; not used (sign differs between prefix and infix): {list(rest.index)}")


if __name__ == "__main__":
    main()
