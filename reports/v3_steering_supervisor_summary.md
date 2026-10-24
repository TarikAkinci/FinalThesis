# v3: locating and steering the identity effect (summary for the supervisor)

Covers v3 step 1 (per-head attention), run B (place-name controls, head profiling) and run C (steering). Both models: Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct. All numbers checked against the result CSVs. Detail reports: `v3_step1_explained.md`, `v3_runB_results.md`, `v3_runC_results.md`, `v3_identity_vs_baseline.md`.

## In short

1. A handful of attention heads in the middle layers carry the region identity effect **causally** (Llama layers 8 to 13, Qwen 13 to 17). Random heads in the same layers do not.
2. The identity information reaches the answer **indirectly**: it is absorbed by the tokens after the phrase, not read directly at the answer position.
3. Steering these heads (PASTA) changes the answer in a **dose-dependent** way. In Llama, the "dampener" heads remove the identity-driven over-flagging specifically and halve the distance to the real raters.
4. **But a constant downward shift does just as well.** By the agreed success criterion, steering's benefit is almost entirely recalibration. No method makes the model follow *which* texts a group finds more offensive.
5. The heads are **not region-specific**. Steering them also separates gender and religion phrases from content-free ones, which prompting alone never did. They respond to "a social group is named", not to a particular group. (This corrects a statement in the run C report, see point 9.)

## 1. Where v2 left us

- Region framing ("You are someone from X.") pushes both models toward "offensive" beyond same-length content-free phrases, in all 10 prompt formats (Llama +1.6, Qwen +2.8 log-odds, prefix). Gender, religion and bare identity words don't.
- The models already over-flag (unframed, about 80% of texts flagged vs 22% majority-offensive), and framing moves them further from the real raters (distance 0.35 → 0.51 Llama, 0.40 → 0.68 Qwen).
- So v3 asked: where in the model does the region effect arise, does it matter causally, and can steering do what prompting cannot?

## 2. Design principles of v3

- **Selection and testing on different texts.** The 500 texts are split into A (250, used to pick heads) and B (250, held out). Every selection rule was written to a file before split B was touched.
- **Four regions** for runs B and C: Sub-Saharan Africa (strongest in both models), Arab Culture, Western Europe, Latin America.
- **Three prompt formats:** prefix (identity before the text), infix (between text and question), target ("offensive *for* X?").
- **The phrase is the statistical unit** for content claims. A phrase is identical on all texts, so item-level p-values are far too optimistic. Effects are judged against **nulls built from interchangeable phrases**.

## 3. Step 1: correlational per-head analysis (CPU, split A)

**Method.**
- For every head (Llama 32 × 32 = 1,024, Qwen 28 × 28 = 784), the attention from the answer position to the identity phrase, recorded in v2. Both raw and value-weighted (attention × size of what the token passes on, Kobayashi et al. 2020).
- **Sink heads removed:** heads with more than 95% of attention on the first token or template (302 in Llama, 104 in Qwen).
- **Content test:** attention *per token* of the phrase, compared with a line through the 8 placebos (attention vs phrase length). A region counts if it sits outside the placebos' scatter around that line. FDR-corrected over heads.
- **Behaviour link:** across texts, does extra attention to the region (vs its placebos) go with an extra shift in the answer? Partial Spearman, controlling for text length and baseline. Text length had produced a spurious link in almost every head: longer texts mean less attention for everything and a smaller effect.

**Result.** 34 candidate heads in Llama, 108 in Qwen, mostly middle layers.

**Interpretation.** Raw attention to a phrase mostly tracks its length (typical rank correlation 0.5 across conditions). Once length is controlled, a smaller set of heads looks at region phrases specially. But this is correlational, and it can't separate identity from place names.

## 4. Run B1: place-name controls (is it identity or the word?)

**Method.** "Someone *from* X" vs "someone *researching* X" and "someone *mapping* X". Same place tokens, same length, but no belonging. Metric: **place-specific shift** = region minus its two place controls, in log-odds; p-value with the 4 regions as units.

**Result.**

| format | Llama | Qwen |
|---|---|---|
| prefix | +0.73 (p = 0.013) | +0.02 (n.s.) |
| infix | +1.01 (p = 0.003) | +1.17 (p = 0.007) |

**Interpretation.** About half of Llama's region effect is identity (belonging), half the place name. In Qwen's perspective formats it is entirely the place name; only when the identity comes after the text does belonging add something.

## 5. Run B2: causal head profiling (PASTA, split A)

**Method.**
- **PASTA steering** (Zhang et al. 2023): in a chosen head, at every position, multiply attention to the identity phrase by a factor ("strength") and renormalise. Strength 100 means everything else is down-weighted to 1%, i.e. the head looks almost only at the phrase. Strength 0 removes the phrase from that head. Strength 1 changes nothing (checked: exactly 0 difference in every run).
- **Profiling:** steer **one head at a time** at strength 100 on 48 split-A texts, under prefix, infix and target, and record how much the answer changes.
- **Two nulls**, because single-head steering at strength 100 changes almost every answer (with items as the unit, 724 of 1,024 Llama heads look "significant"):
  - placebo-pair null: how far the head separates two interchangeable placebos (stairs vs tea);
  - place-pair null: how far it separates "researching X" from "mapping X".
- A head counts only if its region-vs-placebo **and** region-vs-place separation are both beyond the 99th percentile of these nulls.
- **Reliability:** head rankings in two random halves of the 48 texts agree at Spearman 0.82 to 0.96.

**Result.**

| heads beyond both nulls | prefix | infix | target |
|---|---|---|---|
| Llama | 7 | 2 | 1 |
| Qwen | 4 | 6 | 3 |

- Llama: layers 8 to 13. Qwen: layers 13 to 17 (plus a few early heads with unstable direction). Matches the v1 pilot (layers 9 to 18).
- One head alone moves the region answer by 1 to 2.5 log-odds, about the size of the whole region effect.
- **Two kinds of heads.** *Dampeners*: more attention to the identity *lowers* P(offensive) for the region framing, more than for the place name. *Amplifiers*: raise it.
- **Overlap with step 1 is barely above chance** (Llama 6 of 34, chance about 1). Attention patterns are a poor guide to which heads matter causally. From here on we used the causal selection.

**Frozen for run C** (layer.head): Llama dampeners 12.8, 12.6, 11.23, 8.26, 11.25; amplifiers 13.30, 13.4. Qwen dampeners 14.10, 13.24, 6.0; amplifiers 14.24, 13.3, 13.14, 17.18. Dampeners and amplifiers are steered separately (together they cancel).

## 6. Run C: steering (split B, held out)

**What was steered.**

| arm | heads | steered toward | question |
|---|---|---|---|
| dampeners | frozen set | identity phrase | can we remove the identity push? |
| amplifiers | frozen set | identity phrase | can we strengthen it? |
| text | top 10 from a text profiling on split A | the item text | does looking more at the text reduce over-flagging? |
| activation steering | none (layer 16 / 14) | add or subtract the mean "framed minus unframed" hidden-state direction | comparison method that doesn't use attention |
| upper bounds | all heads | identity removed at the answer position only, or at all positions | how does the information travel? |

- Conditions: the 4 regions, man / woman, Christian / Muslim, their length-matched placebos, the place-name controls, and the unframed prompt.
- Strengths 0, 0.5, 2, 4, 10, 100. Every setting repeated with **3 random head sets** with the same number of heads per layer.

**Metrics.**
- **Distance to real raters** (the success criterion agreed on 10 Oct): mean |P(offensive) − share of that group's raters who rated the text offensive|, over split-B texts with at least 4 such raters. All raters for unframed prompts and text steering. Lower = closer.
- **Shift:** change in log-odds vs the same prompt unsteered.
- **Identity-specific shift:** shift of the identity minus the shift of its length-matched placebos, same setting, same texts. **Place-specific shift:** the same against the place controls.
- **Flag rate** (P > 0.5) and **ranking** (Spearman of P with the group's share), so "closer" can't hide a model that just says "no".
- **Constant-shift baseline:** move every unsteered answer down by one fixed amount in log-odds, fitted on split A. Needs no model internals. Also **steering + shift**, cross-fitted on two halves of split B, to see what steering adds beyond calibration.
- **Relative alignment:** does the identity's effect on a text follow that group's real gap on that text (group share minus all raters' share)? Spearman.

## 7. Results, point by point

**R1. Steering is causal and dose-dependent.** Distance to real raters, regions and gender, prefix:

| strength | Llama selected | Llama random | Qwen selected | Qwen random |
|---|---|---|---|---|
| 1 (unsteered) | 0.50 | 0.50 | 0.67 | 0.67 |
| 4 | 0.46 | 0.50 | 0.63 | 0.66 |
| 10 | 0.41 | 0.50 | 0.59 | 0.66 |
| 100 | **0.22** | 0.50 | **0.43** | 0.66 |

- Flag rate (Llama) drops from 73% to 23%. Removing the dampeners' attention (strength 0) moves the answer the other way (0.52), as it should.
- The selected heads matter; random heads in the same layers don't.

**R2. Llama's dampeners remove the identity push specifically.** At strength 100 (mean of prefix and infix), region prompts drop by 2.5 log-odds, placebo prompts *rise* by 1.7 and place-name prompts by 0.6. Ranking is kept (0.15 → 0.18). In Qwen everything drops (regions -5.0, place names -3.4, placebos -1.5): only partly specific, consistent with run B, where Qwen's effect was mostly the place name.

**R3. A constant shift does as well.** Distance after a constant shift: Llama 0.19, Qwen 0.22, at least as good as the best steering. Steering on top of the shift gains at most 0.015 to 0.022 (Llama dampeners prefix, Qwen text steering); everything else changes it by about 0.01 either way. The models over-flag, so anything that pushes answers down "improves" the criterion.

**R4. Text steering stops the detector from flagging.** It reaches a low distance (0.20) by lowering every prompt by 5 to 6 log-odds, unframed included. Only 0 to 3% of texts remain flagged, and ranking gets worse (Llama 0.15 → 0.11, Qwen ~0). Not better detection.

**R5. Amplifiers work in Qwen, and in Llama only relative to placebos.** Qwen: regions +4.1 log-odds, distance grows. Llama: regions barely move in absolute terms (-0.1), but placebos drop by 1.6, so the identity-specific gap still widens by about +1.5.

**R6. Activation steering is weaker.** Removing the framing direction lowers Llama's region prompts by 1.4 log-odds and leaves placebos unchanged (identity-specific), but the distance only drops to 0.38 (vs 0.22 with dampeners). In Qwen it does nothing at the chosen layer.

**R7. No method creates alignment with the real groups.** Relative alignment stays between -0.04 and +0.03 for every method and strength. That matches v2 (0 of 240 tests for prompting).

**R8. The identity reaches the answer through intermediate tokens.** Removing attention to the phrase at the answer position only removes 5 to 19% of the region effect in Llama and about 0% in Qwen. At all positions it removes 89 to 98% (Llama) and 94% (Qwen, prefix). After full removal, region, placebo and place prompts give identical answers. This is why steering has to act at every position, as PASTA does.

**R9. The heads are identity-general, not region-specific (correction).** The run C report said the region heads "lower gender and religion much less than regions" and that "steering did not create an identity effect where prompting had none". That compared absolute shifts. Measured the way we judge identity everywhere else (against length-matched placebos under the same steering, with phrases as units):
- Each of the 4 gender and religion phrases separates from its two exact-length placebos (runner, cyclist) by far more than those two placebos separate from each other. Llama dampeners, strength 10, prefix: -0.9 to -1.8 log-odds vs a placebo-pair gap of 0.10. Strength 100: -2.7 to -3.3 vs 0.35. Llama amplifiers, infix, strength 10: +1.1 to +1.3 vs 0.22. Qwen amplifiers, prefix, strength 10: +1.4 to +3.3 vs 0.10.
- Random head sets don't do this (within about ±0.6 at strength 10).
- The direction follows the head type, and the size is about 40 to 60% of the region effect.
- Exceptions: religion under Llama's amplifiers (prefix) and gender under Qwen's dampeners (prefix) don't separate. At strength 100 Qwen's random sets also become noisy (up to ±4.6), so strength 10 is the cleaner evidence there.

So steering *does* produce a gender and religion effect that prompting alone does not, but it isn't a group *perspective* effect. In absolute terms, gender and religion prompts barely move (-0.7 and -0.5 under Llama's dampeners); the placebos move instead. The heads distinguish "a social group is named" from "a hobby is named". And this can't be checked against real raters: D3CODE has no religion, and gender groups show no reliable text-level difference.

## 8. What it means for the thesis

- **RQ3 answer.** The region identity effect is carried causally by a few middle-layer heads, through intermediate tokens. Attention steering on these heads can switch it off or turn it up, specifically and dose-dependently. Attention *patterns* alone would not have found them; causal profiling with matched nulls did.
- **The constructive result** is Llama's dampener heads: they remove an identity bias (regions down, placebos not), keep ranking, cut over-flagging and halve the distance to real raters. The honest description is "removes an identity-induced bias", not "makes the model judge like the group".
- **Calibration vs alignment.** By the success criterion steering succeeds, but so does a constant shift. The criterion mostly measures calibration, because the models over-flag. Steering adds at most about 10% beyond calibration.
- **Why alignment can't work here.** The follow-up analysis shows the data has almost no text-level group signal: per-text group gaps don't replicate between random halves of the same group's raters (split-half 0.00 to 0.05, same as random pseudo-groups). Only small average strictness offsets exist (regions ±6 points; gender n.s.). Predicting every group with the all-rater average (distance 0.10) beats every model setting. Llama does get the order of the regional offsets right (rho 0.81, n = 8), but with twice the spread.
- **One-line story:** demographic prompting injects a model-made group effect instead of recovering real group perspectives; attention steering can locate and switch off that effect, but it cannot create alignment where the data has almost no group-specific signal.

## 9. Caveats

- Group shares come from at least 4 raters per group and text (often 4 to 7), so they are noisy. That limits every item-level alignment test.
- The constant shift is fitted on split A with the same format. A real deployment would need such labelled data too. It is a reference point, not a free alternative.
- Strength 100 is extreme. Most effects need strength 10 or more, and at 100 random heads also start to move things (especially in Qwen).
- Batched GPU runs differ from single runs by up to about 0.5 log-odds (bf16), so steered vs unsteered are always compared within the same batch.
- Only 2 phrases each for gender and religion; single 0/1 answer; English only; borderline texts; two 7 to 8B models.
- Target ("offensive for X?") uses partly different heads, and steering does nothing useful there. It's a topic-relevance question, not perspective.

## 10. Questions for discussion

1. Is "removes an identity bias, but no better than recalibration by the agreed criterion" an acceptable framing for the steering chapter?
2. Should the identity-general finding (R9) be presented as a result in its own right ("the heads encode *that* a social group is named, not *which*")?
3. Optional extra: profile heads with the groups' own ratings as the target (split A) to give alignment its fairest chance. Given the near-zero group signal in the data, I'd expect little.
