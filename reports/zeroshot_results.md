# Zero-shot baseline: D3CODE + Measuring Hate Speech

Llama-3.1-8B-Instruct, fixed minimal zero-shot prompt (no persona, no few-shot), n=500 per dataset. D3CODE sampled from the frozen test split; MHS binarized at `hate_speech_score > 0.5` (dataset's own convention).

| Dataset | Accuracy | Precision | Recall | F1 | True positive rate | Predicted positive rate |
|---|---|---|---|---|---|---|
| D3CODE | 0.284 | 0.137 | 1.0 | 0.242 | 0.114 | 0.830 |
| MHS | 0.412 | 0.327 | 1.0 | 0.493 | 0.286 | 0.874 |

Naive "always predict not-offensive" baseline accuracy: **0.886** (D3CODE), **0.714** (MHS) — beats Llama on raw accuracy, purely from class imbalance.

## Interpretation

Recall of 1.0 on both datasets + predicted-positive-rate of 83–87% (vs. true rate of 11–29%) means Llama is flagging most items as offensive regardless of content — it never misses a real positive only because it rarely says "not offensive" at all. Accuracy alone is misleading here (naive baseline beats it); precision/recall/F1 are the real story.

This replicates the **"LLMs over-censor"** finding from the lit review — systematic bias toward labeling ambiguous/non-offensive content as offensive — across two independent datasets, under a prompt held fixed with no persona/few-shot influence, so it isn't a prompt-engineering artifact.

Frame as the zero-shot floor: the open question is whether disagreement-aware modeling shifts *where* the model over-predicts, not whether it can classify at all.
