"""
Second selection stage, added after the v3 disagreement + placebo study
showed entropy-selected items are heavily ceiling-concentrated at baseline
(95 of 100 items already predicted "offensive" with median confidence
0.999). That leaves almost no room for any prompt perturbation to push a
prediction FURTHER toward "offensive" -- nearly all observable movement is
downward, which confounds the demographic-vs-placebo probability-shift
comparison regardless of what's actually in the prompt.

This runs the plain baseline prompt (no demographic/placebo prefix) for
every entropy-qualifying item in the seeded 500-item eval sample, keeps only
those whose baseline P(yes) falls in [MIN_BASELINE_PROB, MAX_BASELINE_PROB]
-- i.e. not already pinned near 0 or 1, so there's genuine room to move in
EITHER direction -- and takes the N_ITEMS highest-entropy items among the
survivors. Saves the result for both attention_disagreement_scaled.py and
attention_prompt_variants.py to load via high_disagreement_items.py's
load_filtered_items(), so both experiments use the identical final item set
computed once rather than each redoing this filtering pass.

Needs a GPU (baseline forward pass for every candidate) but is otherwise
cheap: no attention extraction, just predicted/prob_yes per item.

Run as a batch job -- see select_items_with_baseline_filter_job.sbatch.
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attention_probe import build_prompt
from high_disagreement_items import rank_by_entropy, FILTERED_OUT_PATH, N_ITEMS

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
MAX_LENGTH = 1024
MIN_BASELINE_PROB = 0.02
MAX_BASELINE_PROB = 0.98

if __name__ == "__main__":
    candidates = rank_by_entropy()
    print(f"{len(candidates)} entropy-qualifying candidates from the seeded 500-item eval sample "
          f"(entropy range {candidates.entropy.min():.3f}-{candidates.entropy.max():.3f})")

    print(f"\nLoading model: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    )
    model.eval()

    zero_id = tok.encode("0", add_special_tokens=False)[0]
    one_id = tok.encode("1", add_special_tokens=False)[0]

    baseline_pred, baseline_prob = [], []
    with torch.no_grad():
        for i, (_, row) in enumerate(candidates.iterrows()):
            prompt = build_prompt(tok, None, row["text"])
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH,
                         add_special_tokens=False).to(model.device)
            logits = model(**inputs).logits[0, -1]
            baseline_pred.append("1" if logits[one_id] > logits[zero_id] else "0")
            _, prob_one = torch.softmax(logits[[zero_id, one_id]].float(), dim=0)
            baseline_prob.append(prob_one.item())
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(candidates)}")

    candidates["baseline_predicted"] = baseline_pred
    candidates["baseline_prob_yes"] = baseline_prob

    print(f"\nBaseline P(yes) distribution over all {len(candidates)} candidates:")
    print(candidates["baseline_prob_yes"].describe())

    in_band = candidates[(candidates.baseline_prob_yes >= MIN_BASELINE_PROB) &
                          (candidates.baseline_prob_yes <= MAX_BASELINE_PROB)]
    print(f"\n{len(in_band)} of {len(candidates)} candidates have baseline P(yes) in "
          f"[{MIN_BASELINE_PROB}, {MAX_BASELINE_PROB}]")

    final = in_band.sort_values("entropy", ascending=False).head(N_ITEMS).reset_index(drop=True)
    if len(final) < N_ITEMS:
        print(f"WARNING: only {len(final)} items available in-band, short of the {N_ITEMS} target "
              f"-- every entropy-qualifying, in-band candidate is included, this isn't a bug, "
              f"it's the actual size of the intersection of 'high disagreement' and 'room to move'.")

    os.makedirs(os.path.dirname(FILTERED_OUT_PATH), exist_ok=True)
    final.to_csv(FILTERED_OUT_PATH, index=False)
    print(f"\nSaved {len(final)} items to {FILTERED_OUT_PATH}")
    print(f"entropy range: {final.entropy.min():.3f}-{final.entropy.max():.3f}")
    print(f"baseline_prob_yes range: {final.baseline_prob_yes.min():.3f}-{final.baseline_prob_yes.max():.3f}")
    print(final["category"].value_counts())
