---
geometry: margin=2.5cm
fontsize: 11pt
colorlinks: false
---

# Thesis Proposal

**Does the Perspective Register? Probing and Steering Demographic Framing in LLM Offensiveness Judgements**

## Abstract

Even though offensive language detection is often treated as a single-label classification, human judgements of offensiveness differ systematically across cultures and demographic groups. Large Language Models (LLMs) approximate the average human rating but are overconfident, especially on items with high disagreement rates. We ask whether demographic framing in the prompt changes what a model does internally or only what it appears to do from an outsider’s perspective. Using D3CODE, which contains ratings from eight cultural regions, Llama-3.1-8B-Instruct as well as Qwen-2.5-7B, we (i) measure where attention goes at the decision token when a demographic perspective is added, (ii) compare real framings against content-free placebo framings under various prompt structures, and (iii) steer attention toward the framing tokens and test whether this moves predictions toward the framed group’s real ratings [Placeholder for now, until we have the steering results]. Real framings push predictions toward “offensive” while placebo pushes the model in the exact opposite direction, with a comparable size, which hides the actual reversal if only observed as an absolute value. The phrase draws attention from the task instruction rather than from the text being judged and the shifts do not track the real groups’ item-level ratings [this part may be changed after further results]. These results motivate the next stage: steering attention toward the perspective tokens to test whether the demographic signal can influence the judgement, rather than merely accompanying it.

## Introduction

**P1 (Introduce the problem):** Offensiveness is subjective. Single-label gold standards erase real variation. D3CODE (Davani et al.) collects ratings from raters in 21 countries across 8 regions. Demographics alone explain only a small share of rating variance (Hu and Collier), so the interaction with the text matters.

**P2 (LLMs in this role):** Models are accurate on unanimous items and overconfident on low-agreement ones (Lu et al.). They collapse to a flattened consensus, and gain little from persona prompts (Kamruzzaman et al.). Our own check on 500 D3CODE items [and more later] agrees: the model tracks the human average (Spearman rank correlation metric = 0.65), yet its confidence is 0.93 on the most disagreed third versus 0.77 on the most agreed third.

**P3 (Where the gap is):** Since most of prior work in this field is rather behavioural, it rarely asks whether a model uses the demographic cue internally or what effects it has and it very rarely includes a content-free control across different prompt structures. [Mention some of the findings of such behavioural papers like:] Prompt-format effects such as recency bias (Zhao et al.) mean any inserted sentence can move a prediction. Comparing only the absolute size of the shift makes a placebo look as strong as a real framing, so the direction of the shift has to be analysed explicitly.

**P4 (What we do):** [Present all the results and findings] Attention probing at the decision token with the attention-sink caveat (Xiao et al.), placebo and token-matched controls, multiple prompt structures, signed effect analysis, etc. Planned ahead: attention steering (PASTA) and an activation-steering baseline.

**P5 (Contributions and Outline)**

## General Methodology used in our Experiments

Data: D3CODE, seeded 500-item evaluation sample, group-level rating distributions by Region and Gender. Models: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct. Probing: attention at the last token, sink-excluded shares, per layer and per head. Controls: various placebo phrases, token-matched pairs, prefix, reworded, embedded and suffix structures. Steering: PASTA-style reweighting of attention to the perspective span, evaluated by movement of P(yes) toward the target group's mean rating.

## Preliminary evidence

| Finding | Basis |
|:----------------------------------------------------------------------|:------------------------------|
| Model tracks human average (rho 0.65) but is more confident on contested items (0.93 vs 0.77; rho -0.40 between model confidence and human agreement) | 500 seeded items, zero-shot |
| About 80% of attention goes to the sink; the phrase takes share from the instruction, not from the item text | 100 items |
| Real framings shift P(yes) up (+0.05 to +0.11); the placebo shifts it down (-0.08) | 100 items, prefix and reworded |
| The placebo lowers model confidence; real framings do not | Change in prediction entropy |
| Shifts do not track item-level group gaps (all abs rho below 0.2, p above 0.2) | 100 items; regions only n=36 with enough raters |
| Suffix position collapses all conditions downward | 100 items |

Looking only at the absolute change in P(yes), a content-free placebo looks about as strong as a real framing, which suggests the demographic phrase is just noise. Adding the sign shows the opposite: the placebo moves predictions toward "not offensive", and by an amount that rivals or exceeds most of the real framings, which all move toward "offensive". Two real effects of similar size and opposite direction look identical under the absolute value.

| Structure | Condition | Mean signed shift | Mean absolute shift | Items up / down |
|---|---|---|---|---|
| prefix | man | +0.063 | 0.098 | 71 / 23 |
| prefix | woman | +0.050 | 0.116 | 59 / 35 |
| prefix | Western Europe | +0.098 | 0.126 | 77 / 20 |
| prefix | Arab Culture | +0.111 | 0.165 | 75 / 25 |
| prefix | **placebo** | **-0.078** | 0.126 | 31 / 67 |
| reworded | man | +0.045 | 0.096 | 65 / 31 |
| reworded | woman | +0.017 | 0.112 | 50 / 46 |
| reworded | Western Europe | +0.076 | 0.118 | 70 / 26 |
| reworded | Arab Culture | +0.087 | 0.155 | 69 / 29 |
| reworded | **placebo** | **-0.112** | 0.149 | 20 / 75 |
