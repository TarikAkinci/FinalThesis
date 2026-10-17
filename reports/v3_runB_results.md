# Run B results: place-name controls and head profiling

Data: COMA (H200), both models, logs free of errors.
- **B1 (place-name controls):** 8 phrases ("someone researching / mapping <region>") × 10 prompt formats × the same 500 texts. Baselines are identical to v2 to the last digit.
- **B2 (PASTA profiling):** every head steered on its own toward the region phrase, on 48 texts from the selection half (split A), under prefix, infix and target.

Tables: `results/v2/analysis/v3_region_vs_place.csv`, `results/v3/analysis/profiling_*.csv`.

## 1. How much of the "region effect" is just the place name?

We compared "You are someone **from** X" with "You are someone **researching** X" and "... **mapping** X". The place tokens are identical and the length is identical; only the belonging is gone. Values below are log-odds of "offensive", region minus its two place controls, averaged over the 4 regions. The p-value treats the phrase as the unit (4 regions).

| format | Llama | p | Qwen | p |
|---|---|---|---|---|
| prefix | +0.73 | 0.013 | +0.02 | 0.92 |
| reworded / imagine / consider | +0.85 to +0.91 | ≤ 0.02 | -0.83 to -0.01 | mixed |
| infix | +1.01 | 0.003 | +1.17 | 0.007 |
| suffix | -0.05 | 0.67 | -0.02 | 0.90 |
| embedded | +0.45 | 0.07 | +1.16 | 0.016 |
| target | +0.75 | 0.11 | +5.09 | 0.02 |
| context / context_long | +0.76 / +0.46 | 0.06 / 0.15 | -0.38 / -0.05 | 0.03 / 0.70 |

- **Llama.** Roughly half of its region effect is a real identity effect. Being *from* the region pushes toward "offensive" by +0.7 to +1.0 log-odds more than merely naming it, consistently across the perspective formats and infix and for all 4 regions. The other half comes from the place name: "someone researching Sub Saharan Africa" already raises the log-odds by +1.5 under prefix.
- **Qwen.** In the perspective formats (the prefix family and context), the region effect is **entirely the place name**. "Someone researching Western Europe" moves Qwen as much as "someone from Western Europe", and in some wordings even more. A real identity part only appears when the identity comes after the text (infix, embedded). In target the gap is large, but trivially so: "offensive *for* someone mapping Latin America?" gets "no".
- This changes how v2 should be described. "Region is the only identity that matters" holds for Llama, but only half of it is identity. For Qwen in the perspective formats it is a place-name effect.

## 2. Profiling: which heads carry the region identity causally?

**Is the measurement reliable?** Yes. On the GPU, batching shifts single answers by up to about 0.1 to 0.5 log-odds through bfloat16 rounding. Each head's effect is measured within the same batch, though, and the head ranking replicates between two random halves of the 48 texts at 0.82 to 0.96 (Spearman, all models and formats).

**The honest test.** Steering a single head by a factor of 100 changes the answer for almost any phrase, and with items as the unit almost every head looks "significant" (724 of 1,024 in Llama). So each head was judged against two nulls built the same way:
- how far it separates two interchangeable neutral phrases ("stairs" vs "tea");
- how far it separates two interchangeable place phrases ("researching X" vs "mapping X").

A head counts only if its region-vs-placebo **and** region-vs-place-name separation are both beyond the 99th percentile of those nulls.

| | prefix | infix | target |
|---|---|---|---|
| Llama heads passing both | 7 | 2 | 1 |
| Qwen heads passing both | 4 | 6 | 3 |

- **Few heads, all in the middle layers.** Llama's are in layers 8 to 13, Qwen's in layers 13 to 17 (plus a few in layers 0 and 6 whose direction is unstable). This matches step 1 and v1 (layers 9 to 18).
- **Large single-head effects.** Turning one of these heads up changes the region answer by 1 to 2.5 log-odds, about the size of the whole region effect.
- **Two kinds of heads.**
  - *Dampeners:* turning them up toward the identity **lowers** P(offensive) for the region framing, more than for the place name alone. Llama: layer 12 heads 8 and 6, layer 11 heads 23 and 25, layer 8 head 26. Qwen: layer 14 head 10, layer 13 head 24, layer 6 head 0.
  - *Amplifiers:* turning them up **raises** it. Llama: layer 13 heads 30 and 4. Qwen: layer 14 head 24, layer 13 heads 3 and 14, layer 17 head 18.
- **Formats agree only partly.** Head rankings correlate at 0.45 between prefix and infix, and only 0.15 to 0.33 with target. Target again behaves like a different question.
- **Correlational vs causal selection agree only weakly.** The step-1 candidates (picked by attention patterns) overlap with the top profiling heads a little above chance (Llama 6 of 34, chance about 1; Qwen 27 of 108, chance about 15). Attention patterns alone are a poor guide to which heads matter, a known point in the interpretability literature. From here on we use the causal selection.

## 3. Frozen for run C

Rule (`results/v3/analysis/heads_rule.txt`): heads passing both nulls in at least one format, grouped by the direction of their effect, which has to agree in prefix and infix. Ranked by effect size. Dampeners and amplifiers are steered separately, because together they would cancel out.

| model | dampeners | amplifiers |
|---|---|---|
| Llama | 12.8, 12.6, 11.23, 8.26, 11.25 | 13.30, 13.4 |
| Qwen | 14.10, 13.24, 6.0 | 14.24, 13.3, 13.14, 17.18 |

(layer.head)

**Why the dampeners matter for the success criterion.** Framed answers are too high compared with the real raters (v2: distance 0.51 for Llama and 0.68 for Qwen, against 0.35 and 0.40 unframed). Heads whose amplification lowers P(offensive) specifically for identity prompts are the most direct route to "closer to the real people". Run C tests exactly that on the held-out half.

## 4. Still open

- The per-head *attention* files of the place run didn't arrive on the laptop. With them, the correlational step-1 candidates could also be checked against the place names. Since we now select heads causally, this is no longer on the critical path. It would only matter for the thesis's description of the correlational step.
