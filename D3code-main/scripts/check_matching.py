"""
Tokenizer-only check that the v2 design is length-equal across models: every
(variant, condition) demographic span must have the same token length under
every model's tokenizer, and every demographic must have exact-length placebos
in every variant under every tokenizer. Runs on a laptop (no weights needed).

  python check_matching.py                       # Llama and Qwen
  python check_matching.py --models a/b c/d      # any set of models
"""
import argparse

import pandas as pd
from transformers import AutoTokenizer

import master_grid as mg

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+",
                default=["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct"])
a = ap.parse_args()

tables = {}
for m in a.models:
    tok = AutoTokenizer.from_pretrained(m)
    matches, lengths = mg.build_matching(tok)           # aborts if a demographic has no exact placebo
    tables[m] = lengths
    print(f"{m}: every demographic has an exact-length placebo in all {len(mg.VARIANTS)} variants")

first = a.models[0]
print(f"\nDemographic span length per variant ({first}):")
print(tables[first].to_string())
bad = []
for m in a.models[1:]:
    diff = tables[m] != tables[first]
    for cond, variant in zip(*diff.values.nonzero()):
        bad.append((tables[m].index[cond], tables[m].columns[variant],
                    tables[first].iat[cond, variant], tables[m].iat[cond, variant], m))
if bad:
    print("\nLENGTH DIFFERS BETWEEN MODELS:")
    print(pd.DataFrame(bad, columns=["condition", "variant", first, "other", "model"]).to_string(index=False))
    raise SystemExit(1)
print(f"\nOK: all {tables[first].size} (condition, variant) span lengths are identical across {len(a.models)} models.")
