# Proposal sketch (working draft)

Status: rough sketch for discussion, not final text. Numbers marked [update] should be refreshed once the master grid finishes (8 regions, 3 placebos, token-matched pairs, longer contexts, Qwen). Citations are taken from the paper summaries in this folder; check bibliographic details before submission.

---

## 1. Which direction to take

**Recommended: interpretability of demographic framing, with steering as the intervention.**

The story your data already supports:

1. Offensiveness is contested across groups, and LLMs are confident exactly where humans are not.
2. Telling the model "you are X" does change its judgments, but only in a way that a content-free sentence also changes them (in the opposite direction), and the phrase receives little attention.
3. The changes do not line up with what the real groups actually rated.
4. So the open question is whether the perspective can be made to matter internally. Steering attention or activations toward the perspective tokens is the natural test, and it is the method your supervisor pointed at (PASTA).

Two alternatives, so you can choose consciously:

| Option | Idea | Upside | Downside |
|---|---|---|---|
| A (recommended) | Probe, control, then steer | Uses everything done so far, matches supervisor's staging | Steering results are still ahead of you |
| B | Calibration to disagreement (make model uncertainty track human disagreement) | Closest to existing literature, easy metrics | Little interpretability, less original |
| C | Methodology paper on placebo-controlled persona evaluation | Your most original results are here (signed effects, placebo, ceiling confound) | Weaker link to the steering plan |

A good plan is A as the main line, with C's methodology as the first contribution.

## 2. Working titles

1. Does the Perspective Register? Probing and Steering Demographic Framing in LLM Offensiveness Judgments
2. Confident Where Humans Disagree: A Placebo-Controlled Look at Demographic Prompting in Offensive-Language Detection
3. Placebo-Controlled Probing of Demographic Framing in LLM Offensiveness Judgments

## 3. Page-one sketch

### Abstract (draft, about 170 words)

Offensive language detection is usually framed as single-label classification, yet human judgments of offensiveness differ systematically across cultures and demographic groups. Large language models approximate the average human rating but are overconfident on the items humans contest. We ask whether demographic framing in the prompt changes what a model does internally, or only what it appears to do. Using D3CODE, which contains ratings from eight cultural regions, and Llama-3.1-8B-Instruct, we (i) measure where attention goes at the decision token when a demographic framing is added, (ii) compare real framings against content-free placebo framings under several prompt structures, and (iii) steer attention toward the framing tokens and test whether this moves predictions toward the framed group's real ratings. The framing phrase draws attention from the task instruction rather than from the text being judged. Real framings push predictions toward "offensive" while a placebo pushes them the other way, with a comparable size, so that unsigned effect sizes make the two look alike and hide the reversal, and the shifts do not track the real groups' item-level ratings [update]. [PLACEHOLDER, replace after the steering experiments: Steering attention toward the framing tokens changes P(yes) by [X] on average and [improves / does not improve] agreement with the framed group's ratings ([metric] = [value] versus [value] for plain prompting).] [PLACEHOLDER closing sentence, adapt to the steering outcome: These results show that the demographic signal [can / cannot] be made to influence, rather than merely accompany, the judgment.]

### 1 Introduction (paragraph outline)

- **P1, the problem.** Offensiveness is subjective. Single-label gold standards erase real variation. D3CODE (Davani et al.) collects ratings from raters in 21 countries across 8 regions. Demographics alone explain only a small share of rating variance (Hu and Collier), so the interaction with the text matters.
- **P2, LLMs in this role.** Models are accurate on unanimous items and overconfident on low-agreement ones (Lu et al.), collapse to a flattened consensus, and gain little from persona prompts (Kamruzzaman et al.).
- **P3, the gap.** Prior work is behavioural. It rarely asks whether a model uses the demographic cue internally, and it rarely includes a content-free control. Prompt-format effects such as recency bias (Zhao et al.) mean any inserted sentence can move a prediction. Comparing only the absolute size of the shift makes a placebo look as strong as a real framing, so the direction of the shift has to be analysed explicitly.
- **P4, what we do.** Attention probing at the decision token (with the attention-sink caveat, Xiao et al.), placebo and token-matched controls, multiple prompt structures, signed effect analysis. Planned: attention steering (PASTA) and an activation-steering baseline. Motivating evidence from our own check on 500 D3CODE items: the model tracks the human average (Spearman 0.65), yet its confidence is 0.93 on the most contested third of items versus 0.77 on the most agreed third.
- **P5, contributions and outline.**

### Research questions

- **RQ1.** Is the effect of a demographic framing distinguishable from the effect of any inserted sentence of matched length?
- **RQ2.** Where does the framing register (layers, heads), and does attention to it predict the behavioural shift?
- **RQ3.** Can steering toward the perspective tokens make the model's judgments track the framed group's real ratings better than plain prompting?

### Contributions

1. A controlled protocol for evaluating demographic prompting: placebo, exact token matching, several prompt positions, signed rather than unsigned effects, and a baseline-probability band to avoid ceiling effects.
2. Empirical findings on Llama-3.1-8B [update: plus Qwen2.5-7B, all 8 regions, 3 placebos].
3. A steering test of whether the perspective can be made to matter internally.

## 4. Method in five lines

Data: D3CODE, seeded 500-item evaluation sample, group-level rating distributions by Region and Gender. Models: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct. Probing: attention at the last token, sink-excluded shares, per layer and per head. Controls: three placebo phrases, token-matched pairs, prefix, reworded, embedded and suffix structures. Steering: PASTA-style reweighting of attention to the perspective span, evaluated by movement of P(yes) toward the target group's mean rating.

## 5. Preliminary evidence (what you can already state)

| # | Finding | Basis | Status |
|---|---|---|---|
| 1 | Model tracks human average (rho 0.65) but is more confident on contested items (0.93 vs 0.77; rho -0.40 between model confidence and human agreement) | 500 seeded items, zero-shot | Solid for one prompt and model. The model answers "yes" to about 83% of items, which contributes to the pattern |
| 2 | About 80% of attention goes to the sink; the phrase takes share from the instruction, not from the item text | 100 items | Solid |
| 3 | Real framings shift P(yes) up (+0.05 to +0.11); the placebo shifts it down (-0.08) | 100 items, prefix and reworded | Placebo is one sentence [update]. Woman is not significant in reworded (p=0.52) |
| 4 | The placebo lowers model confidence; real framings do not | Change in prediction entropy | New, same caveat as row 3 |
| 5 | Shifts do not track item-level group gaps (all abs rho below 0.2, p above 0.2) | 100 items; regions only n=36 with enough raters | New, underpowered for regions. At group level, Arab raters rate higher on these items (+0.12) and Arab framing pushes up most, which is suggestive only |
| 6 | Suffix position collapses all conditions downward | 100 items | Solid, explained by recency bias |

### The signed-versus-unsigned point (your most original finding, state it explicitly)

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

The placebo's signed shift is larger than 2 of the 4 real framings in prefix (it exceeds man and woman, not Western Europe or Arab Culture) and larger than all 4 in reworded. Caveats: the placebo is one sentence so far [update], and gender_woman is not significant in reworded (p=0.52).

Not yet supported: per-layer or per-head claims on the current item set, religion, other regions, length-matched identity phrases, any second model.

## 6. Questions to settle with your supervisor

1. Main axis: regions (D3CODE's cultural framing) or gender, or both?
2. Is a second model required, or is Llama plus one replication enough?
3. What is the steering success criterion: shift toward the target group's mean rating, or toward its full rating distribution?
4. Scope: a one-page proposal now, or a short paper draft?

## 7. Honest limits to state up front

One dataset, two models, one prompt template family. Single placebo phrase so far. Per-item p-values for phrase-level attention effects overstate independence, because the phrase is identical across items. Attention is a correlational signal, not proof that the model uses the phrase.
