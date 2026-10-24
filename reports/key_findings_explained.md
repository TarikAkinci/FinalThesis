# Key findings, explained for a technical audience

Status: all experiments done (v1 pilots, v2 = run A, v3 = runs B and C, follow-ups). Numbers below are the current ones, checked against the result CSVs. Sources are given in brackets.

## Background you need to follow the results

- **Data.** D3CODE: 4,590 short texts (from Jigsaw), each rated for offensiveness by about 30 raters from 21 countries in 8 cultural regions, with rater gender, age and moral-foundation scores. So we know, per text, how each region or gender actually judged it. Rater religion is not recorded.
- **Models.** Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct, open weights, so we can read and change their internals.
- **Task and measurement.** Prompt: `Text: "..." Is this text offensive? Answer with only 0 (no) or 1 (yes).` No text is generated. We run one forward pass and read the model's probability of "1" vs "0" directly from the output logits (computed in fp32). The main outcome is the change in **log-odds** of "offensive" compared with the same text without any framing. +1 log-odds means the odds of "offensive" are multiplied by about 2.7. Log-odds don't saturate near 0 and 1, so texts with different starting points are comparable.
- **Framing.** We add an identity ("You are someone from the Arab Culture region.", "You are a woman.", "You are a Muslim.") in 10 prompt formats: before the text, after it, between text and question, as "Assume / Imagine / Consider that you are...", embedded ("As a woman, is this offensive?"), target ("Is this offensive *for* a woman?"), and with an extra sentence about background.
- **Placebos.** The key control. Content-free personas of the same token length ("You are a runner.", "You are someone who prefers tea over coffee."). Any inserted sentence moves an LLM's answer, so an identity only counts if it does more than a placebo of the same length.
- **Place controls.** "Someone *researching* / *mapping* X" uses the same place name and length as "someone *from* X" but drops the belonging. This separates "identity" from "the model reacts to the word Africa".
- **Design.** 500 texts on which both models are unsure (baseline P between 0.02 and 0.98, so there is room to move both ways). Fully crossed: 22 conditions × 10 formats × 500 texts = 110,000 prompts per model. Texts are split in half: A for choosing things (heads), B held out for testing.
- **Statistics.** A phrase is identical on all 500 texts, so using texts as the unit makes almost anything "significant". For claims about identity *content*, the **phrase is the unit** (demographic phrases vs the 8 placebos), with phrase length as a covariate.
- **Attention and heads.** Each layer has many attention heads (Llama 32 layers × 32 heads, Qwen 28 × 28). A head decides how much each token "looks at" every earlier token. A large share of attention always goes to a few fixed tokens at the start (the "attention sink"), regardless of content.
- **Steering.** PASTA (Zhang et al. 2023): in chosen heads, multiply the attention paid to a chosen span (here the identity phrase) by a factor, then renormalise. Factor 0 removes the phrase from those heads, factor 100 makes them attend almost only to it. Activation steering (comparison method): add or subtract a fixed direction vector in the model's hidden states.

## RQ1: Do the models react to identity, beyond what content-free phrases do?

- **Region: yes, clearly.** Length-adjusted, region framings shift the answer toward "offensive" by +1.6 log-odds (Llama) and +2.8 (Qwen) more than placebos, under the plain prefix format, and significantly in all 10 formats in both models. Average P(offensive) goes from about 0.5 to 0.70 (Llama) and 0.87 (Qwen). [v2_behavioural_analysis §2]
- **Sub-Saharan Africa has the largest effect in both models** (+3.0 and +5.6 log-odds vs a same-length placebo). Even Western Europe and North America push upward. [§2]
- **Gender, religion and bare identity words ("an Arab"): not distinguishable from placebos.** "You are a woman" behaves like "You are a runner". With only 2 phrases per axis power is limited, but the estimates sit inside the placebo spread. [§2]
- **Length is a real confound.** Before adjusting for phrase length, gender looks significant (Llama p = 0.006). After adjusting, it isn't (p = 0.59). Longer content-free phrases push the answer down in every format, and region phrases happen to be the longest. Without length control the conclusions would be wrong in both directions. [behaviour_axis_tests.csv, §3]
- **Half of Llama's "region identity" is the place name.** "Someone from X" beats "someone researching X" by +0.7 to +1.0 log-odds, so about half of the effect is belonging and half is the word itself. In Qwen's perspective formats the effect is **entirely the place name**. [v3_runB §1]
- **Qwen reacts to any persona.** Most placebos alone push Qwen toward "offensive" by +2.3 to +3.8 log-odds ("You are a cyclist." takes it from 0.53 to 0.85). Llama's placebos sit around zero. [§3]
- **Wording of the template doesn't matter, position does.** "You are / Assume / Imagine / Consider" agree almost perfectly (rank correlation 0.90 to 0.99). Putting the identity *after* the question pulls everything down, and in Qwen it even reverses the identity effect. [§4, §5]
- **"Offensive for X" is a different question from "as X".** Under the target format, placebos collapse ("offensive for a runner?" gets "no"), so it measures whether the text is about that group, not perspective taking. Reported separately. [§5]
- **Telling Llama that background matters makes it use gender.** With the extra sentence "Your background and everyday experience shape how you read things.", gender gets an effect (+1.07, p = 0.007). Without it, none. [§5]
- **Human disagreement doesn't predict sensitivity.** Contested texts are not more affected by framing than clear ones. [§7]

## RQ2: Do these shifts match how the real groups judge the texts?

- **No. Framing moves the models away from real people.** Mean distance between the model's P(offensive) and the share of that group's raters who called the text offensive: Llama 0.35 unframed → 0.51 framed, Qwen 0.40 → 0.68. A same-length placebo moves them less. [§8]
- **The models over-flag to begin with.** On all 4,519 texts, unframed: Llama flags 79% and Qwen 85% as offensive, while only 22% are majority-offensive. Recall 0.99, precision about 0.26, but ROC-AUC 0.83: they rank texts reasonably, their threshold is badly off. Framing makes this worse. [v3_identity_vs_baseline §5]
- **No text-by-text alignment.** Do texts that a group finds more offensive than others get a bigger push under that group's framing? 0 of 240 tests significant; mean correlation 0.00 to 0.02. [§8]
- **But Llama knows the average regional strictness order.** Real raters differ slightly by region on average (Arab Culture +6.6 points stricter than overall, Oceania -6.5). Llama's region effects follow that order (rank correlation 0.81 over 8 regions, p ≈ 0.015), but with twice the real spread and everything shifted up. Qwen: 0.05. [v3_identity_vs_baseline §4]

## A finding about the data itself

- **In D3CODE, groups barely differ text by text.** Splitting each group's raters into two random halves, the per-text group differences don't replicate (split-half correlation 0.00 to 0.05; about 0.1 for Sinosphere and Sub-Saharan Africa). Random pseudo-groups of the same size look the same. [v3_identity_vs_baseline §2]
- **The same holds for moral-value groups** (raters split by care, loyalty, purity, ...): small average offsets (high-care raters +5.6 points stricter), no per-text signal. [§6]
- **Disagreement lives mostly inside groups, not between them** (within-region entropy 0.64 vs 0.80 overall). [§6]
- **Practical consequence:** predicting each group by the average of *all* raters (distance 0.10) beats every model setting by a factor of two. There is no reliable group-specific signal for a model to "recover". This matches the literature finding that demographics explain only a few percent of annotation variance.

## RQ3: Where does the effect live in the model, and can steering control it?

- **Attention patterns alone are a weak guide.** Picking heads by how much they attend to the identity (correlational) overlaps only slightly with picking them by what happens when you change them (causal profiling): Llama 6 of 34 vs about 1 by chance. Raw attention to a phrase mostly tracks its length. [v3_runB §2, v3_step1_explained]
- **Causal head profiling finds a few heads in the middle layers.** Each head was steered on its own (PASTA, factor 100) and judged against two nulls: how much it separates two interchangeable placebos, and two interchangeable place phrases. Beyond the 99th percentile of both: Llama 7 heads (layers 8 to 13), Qwen 4 to 6 (layers 13 to 17). The ranking replicates across random halves of the texts (0.82 to 0.96). One head alone can move the region answer by 1 to 2.5 log-odds. [v3_runB §2]
- **Two kinds of heads.** "Dampeners": attending more to the identity *lowers* P(offensive) for region framing. "Amplifiers": raise it. Steered separately, because together they cancel. [v3_runB §3]
- **The identity reaches the answer indirectly.** Removing all attention to the identity *only at the answer position* removes just 5 to 19% of the effect (Llama), about 0% (Qwen). Removing it at all positions removes 89 to 98%. The phrase is first absorbed into the following tokens, which then carry it to the answer. That is why steering must act at every position, as PASTA does. [v3_runC §4]
- **Steering works, causally and dose-dependently.** On held-out texts, steering the frozen dampener heads cuts the distance to real raters from 0.50 to 0.22 (Llama) and 0.67 to 0.43 (Qwen). Random heads with the same layer mix do nothing (0.50, 0.66). [v3_runC §1]
- **Llama's dampeners remove the identity bias specifically.** Region prompts drop by 2.5 log-odds, while placebo prompts rise by 1.7 and place-name prompts barely move. Ranking ability is kept (rank correlation with the group's share 0.15 → 0.18). Flagged texts drop from 75% to 24%. The cleanest intervention in the project. In Qwen it is only partly specific, consistent with its effect being largely the place name. [§2]
- **But a constant downward shift does as well.** Because the models over-flag, anything that lowers P(offensive) reduces the distance. Shifting every answer down by a fixed amount (fitted on split A) gives 0.19 (Llama) and 0.22 (Qwen), as good as or better than any steering. Steering on top of that shift gains at most about 0.02. So by this criterion, steering's benefit is almost entirely **calibration**. [§1]
- **Steering toward the text breaks the detector.** It reaches a low distance (0.20) by answering "no" to nearly everything (0 to 3% flagged), and ranking gets worse. [§2]
- **Activation steering is weaker** than head-targeted attention steering (Llama 0.38, Qwen no effect at the chosen layer). [§2]
- **Steering doesn't create alignment.** After steering, the identity's effect still doesn't follow which texts a group finds more offensive (correlation -0.04 to +0.03, every method and strength). It also doesn't create gender or religion effects: the region heads are region-specific. [§2, §3]

## Methodological lessons worth sharing

- Always compare with **length-matched content-free placebos**; any inserted sentence moves an LLM.
- For claims about phrase content, the **phrase is the statistical unit**, not the item.
- Control **phrase length**: it drives both attention and behaviour.
- Separate **identity from vocabulary** (place controls): "from Africa" vs "researching Africa".
- Select heads **causally** and judge them against **matched nulls**, not item-level p-values. Select on one half of the data, test on the other.
- When a model is miscalibrated, any "closer to humans" metric must be compared with a **trivial recalibration baseline**, plus ranking and flag rate. Otherwise you mistake "says no more often" for "understands the group".

## The story in one line

Demographic prompting injects a model-made group effect instead of recovering real group perspectives. Attention steering can locate that effect in a handful of middle-layer heads and switch it off, but it cannot create alignment where the data has almost no group-specific signal.

## Caveats

- Texts were chosen to be borderline for both models (milder and less contested than D3CODE overall).
- Single 0/1 answer, English only, two 7 to 8B models.
- Only 2 phrases per axis for gender, religion and bare identity, so little power there.
- Per-text group shares come from 2 to 7 raters per region and are noisy; this limits any item-level alignment test, which is partly why the data finding above matters.
- Steering factor 100 is extreme; most of the effect needs factors of 10 or more.
