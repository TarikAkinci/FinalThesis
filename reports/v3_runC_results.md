# Run C results: steering

Data: COMA (H200), both models. Held-out split B: 250 texts that were never used to select heads. Logs are free of errors. The hook's "strength 1 changes nothing" check is exactly 0 in every run.

Tables: `results/v3/analysis/steering_summary.csv`, `steering_vs_trivial.csv`. Figure: `figures/v3/steering_dose_response.png`.

## What was run

Every steering setting was applied to:
- the 4 regions, gender (man / woman) and religion (Christian / Muslim);
- their length-matched placebos and the place-name controls;
- the unframed prompt;

under prefix, infix and target, at strengths 0 (attention to the span removed), 0.5, 2, 4, 10 and 100 (PASTA's default). Each run was repeated with 3 random head sets that have the same layer mix.

| setting | heads | steered toward |
|---|---|---|
| dampeners | Llama 5, Qwen 3 (frozen from run B) | the identity phrase |
| amplifiers | Llama 2, Qwen 4 (frozen from run B) | the identity phrase |
| text | top 10 from a text profiling on split A | the item text |
| activation steering | layer 16 / 14, no heads | adds or removes the average "framing" direction |
| upper bounds | all heads | attention to the identity removed at the answer position only, or everywhere |

Success criterion (agreed with the supervisor): the distance between the model's P(offensive) and the real raters' average gets smaller. For framed prompts the reference is the raters from that region or gender; otherwise it is all raters.

## 1. Main result: steering meets the criterion, but so does a trivial fix

Distance to the real group's average for the framed prompts (regions and gender, prefix):

| | Llama | Qwen |
|---|---|---|
| unsteered | 0.50 | 0.67 |
| dampeners, strength 100 | 0.22 | 0.43 |
| text steering, strength 100 | 0.20 | 0.20 |
| random heads, strength 100 | 0.50 | 0.66 |
| activation steering, framing direction removed | 0.38 | 0.66 |
| **constant downward shift (trivial fix)** | **0.19** | **0.22** |

The constant shift moves every unsteered answer down by the same amount on the log-odds scale. It is fitted on split A and applied to split B. It needs no model internals at all.

- Steering clearly works by the criterion. Dampener and text steering cut the distance by more than half. The effect grows steadily with strength, and random heads in the same layers do nothing. So the selected heads really matter, and the effect is causal.
- **But no steering method beats the constant shift.** The models over-flag (their average P(offensive) is far above the raters' share, which is about 0.2). Anything that pushes the answers down shrinks the distance, and a constant shift does that best.
- So the success criterion, taken alone, can be met by recalibration. It cannot tell "the model now understands the group" apart from "the model now says no more often". The checks below separate the two.

**Does steering add anything on top of the shift?** We fitted a constant shift on half of the held-out texts and evaluated it on the other half, once on the unsteered answers and once on the steered ones (`steering_plus_shift_cv.csv`):
- Llama dampeners at strength 100 (prefix): 0.189 with the shift alone, 0.173 with steering plus shift (-0.015).
- Qwen text steering at strength 100: -0.016 (prefix) and -0.022 (infix).
- Every other setting changes it by at most about 0.01, in either direction.

So steering adds at most about 10% beyond calibration, and only in a few settings. Almost all of its benefit by this criterion is calibration.

## 2. What each method actually does

**Text steering (arm D) makes the models stop flagging.**
- It lowers the log-odds of every prompt by about 5 to 6, including the unframed one. It is not identity-specific.
- At strength 100, only 0 to 3% of texts are still called offensive (before: about 50 to 90%).
- The little ranking ability the models had gets worse: Llama's Spearman correlation with the real share drops from 0.13 to 0.03.
- So the distance shrinks for the wrong reason. Text steering is not better offensiveness detection. It is a model that almost always answers "no".

**Dampener steering (arm C) removes the identity push, specifically.**
- In Llama at strength 100, region prompts drop by 2.5 log-odds, while placebo prompts rise by 1.7 and place-name prompts barely move (+0.6). This is the clearest identity-specific intervention in the whole project.
- In Qwen, regions drop most (-5.0), but placebos (-1.5) and place names (-3.4) drop too. The effect is partly specific, matching run B, where Qwen's region effect was largely a place-name effect.
- It keeps the ranking (Llama prefix: Spearman with the group's share rises slightly, from 0.15 to 0.18). Flagged texts go from 75% to 24% (Llama).
- It works in prefix and infix, the formats the heads were selected on. In target it does nothing or makes things worse. Target is a different question, and run B already showed it uses different heads.

**Amplifiers (arm A) push the other way, as predicted, but weakly.** In Qwen, regions rise by +4.1 log-odds and the distance grows by 0.09. In Llama the two amplifier heads barely move the regions (-0.1). Their single-head profiling effect did not carry over to steering them together.

**Gender and religion (arm B).** The region dampener heads lower gender and religion prompts much less than region prompts in Llama (-0.7 and -0.5, against -2.5). Steering did not create an identity effect where prompting had none. The heads are region-specific.

**Activation steering (comparison method).**
- In Llama, removing the average framing direction lowers region prompts by 1.4 log-odds and leaves placebos unchanged. That is identity-specific, but weaker than the dampener heads (distance 0.38 vs 0.22).
- In Qwen it does almost nothing at the chosen layer.

## 3. Does steering make the identity follow the real groups?

No. The question here is whether, after steering, the identity's effect on a text tracks how differently that group actually rated it (group share minus all raters' share, per text). Across every method and strength the Spearman correlation stays between -0.04 and +0.03. No method creates relative alignment with the real groups. That is the same null result as in v2 for prompting.

## 4. Upper bounds: where does the identity information travel?

| | Llama | Qwen |
|---|---|---|
| identity removed only at the answer position | 5 to 19% of the region effect gone (prefix, infix) | about 0% |
| identity removed at all positions | 89 to 98% gone | 94% (prefix) |

- The answer position barely reads the identity phrase directly. The information is first absorbed into intermediate tokens and only then reaches the answer. This was predicted in the v3 plan, and it is why steering has to act at all positions, as PASTA does.
- After complete removal, region, placebo and place-name prompts all give exactly the same answer. The removal really deletes the content. Any remaining gap to the unframed prompt comes from the format itself (invisible tokens between the text and the question).

## 5. What this means for the thesis

1. **The causal story holds.** A handful of middle-layer heads (Llama layers 8 to 13, Qwen 13 to 17) carry the region identity effect. Steering them changes the answer in a dose-dependent, identity-specific way that random heads in the same layers don't. Attention steering is a working tool for locating and controlling the effect.
2. **"Closer to real people" is reachable, but only as recalibration.** Every method that reduces the distance does so mainly by lowering P(offensive) overall. A constant shift does it as well or better. None of them makes the model follow *which* texts a group finds more or less offensive.
3. **The constructive finding is the dampener heads in Llama.** They remove the identity-driven over-flagging specifically (regions down, placebos not), keep the ranking, and halve the distance to the real group. That is the cleanest intervention in the project. Its honest description: "removes an identity bias", not "makes the model think like the group".
4. **The negative findings are results too.**
   - Steering toward the text breaks detection rather than improving it.
   - Activation steering is weaker than head-targeted attention steering here.
   - No method creates group-specific alignment.

## 6. Caveats

- Distances are measured against small rater groups (at least 4 raters per group and text), so per-text group shares are noisy. That lowers every possible correlation and is one reason item-level alignment is hard to detect at all.
- The constant-shift baseline is fitted on split A with the same prompt format. A real deployment would need such data too. It is a reference point, not a free alternative.
- Strength 100 is extreme. The dose curves show that much of the effect needs strengths of 10 or more.
- Answers are a single 0/1 token, English only. The texts are borderline for both models.

## Suggested next steps (for discussion)

- Report the dampener result as the main constructive steering finding, with the constant-shift comparison and the small gain on top of it (section 1) stated openly.
- Optionally, profile heads with the group's ratings themselves as the target (selected on split A) to give arm C its fairest chance. Given the null relative alignment everywhere, I'd expect little.
