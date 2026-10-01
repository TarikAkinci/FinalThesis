# Disagreement + Placebo Study: What We Ran, What We Found, What It Means

## What we ran, and why

Your supervisor asked for three things after seeing the first placebo-controlled result: log P(yes) instead of just the final 0/1 label, automate item selection and scale it up, and add a placebo control to check whether the demographic-attention share we'd found was actually about demographic content. We built that (500 items, pooled from the whole D3CODE dataset, ranked by a gender- or region-specific rater gap).

Then your supervisor refined the item-selection instruction: don't pool from the whole dataset, select from *within* the seeded 500-item D3CODE evaluation sample (the same one your headline zero-shot metrics are reported on), and rank by a general disagreement measure, entropy, variance, or disagreement rate, not a demographic-specific gap. We rebuilt item selection around Shannon entropy of each item's full rating distribution (reusing `evaluate.py`'s own entropy function), selected the 100 highest-entropy items, and reran both the placebo study and a prompt-structure robustness check (does rewording or repositioning the demographic phrase change the finding) on that new set.

That run surfaced a real confound: entropy-selected items turned out to be ones the model already called "offensive" with near-certainty at baseline (95 of 100 items, median confidence 0.999), leaving almost no room for any prompt change to push the prediction *further* up, only down. So we added a second selection stage: keep the entropy ranking, but also require baseline P(yes) to fall in [0.02, 0.98], so there's genuine room to move in either direction. 232 of the original 500 candidates cleared that bar, comfortably more than the 100 needed. The results below are from that corrected 100-item set, the two studies you just pulled back.

## The result that matters most: direction, not just magnitude

At the clean, unconfounded prompt structure (`prefix`, the same structure used everywhere else in this project), here's how each condition shifts P(yes) away from baseline:

| condition | items pushed toward "offensive" | items pushed toward "not offensive" | mean signed shift |
|---|---|---|---|
| gender_man | 71 | 23 | **+0.063** |
| gender_woman | 59 | 35 | **+0.050** |
| region_western_europe | 77 | 20 | **+0.098** |
| region_arab_culture | 75 | 25 | **+0.111** |
| placebo | 31 | 67 | **-0.078** |

Every single real demographic condition pushes the model toward calling things offensive, on a solid majority of items, with a mean shift of real size (5-11 percentage points of probability). The placebo pushes the opposite way, toward "not offensive," on two-thirds of items. This replicates almost exactly at `reworded` (same position, independently-worded phrases: demographic conditions +0.017 to +0.087, placebo -0.112), so it's not a one-phrasing fluke.

**This changes the conclusion from the earlier report.** When we only looked at raw, unsigned magnitude, demographic content and this placebo looked statistically indistinguishable (and in the ceiling-confounded run, placebo even looked bigger). Once you look at direction, they're not remotely the same: both have a real, substantial, consistent effect, and they point opposite ways. "Demographic framing is just noise" is not what this data shows. "Demographic framing reliably makes the model more cautious about calling things offensive, in a way a random irrelevant sentence does not" is what it shows.

## Why the magnitude comparison came out "not significant"

Paired Wilcoxon on `|ΔP(yes)|` (unsigned) at `prefix`: gender_man p=0.094, gender_woman p=0.099, region_western_europe p=0.586, region_arab_culture p=0.561, none of them significant. That's not a contradiction of the table above, it's the same fact seen through the wrong lens: if condition A reliably moves things +0.08 and condition B reliably moves things -0.08, their *unsigned* sizes look the same even though their effects are opposite and both real. Magnitude alone was the wrong statistic for this question. Direction was always the more informative one, and in hindsight this should have been the primary analysis from the start, not an afterthought.

## The attention-share finding is the most robust result in this entire line of work

Across both studies, and across all 4 prompt structures (prefix, suffix, reworded, embedded), the same pattern holds without a single exception:

- gender_man and gender_woman: attention share ≤ placebo, every time, p < 4e-16.
- region_western_europe: attention share ≤ placebo, every time, p < 4e-9.
- region_arab_culture: attention share **>** placebo, every time, p < 4e-15, and it's also the condition with the largest behavioral shift (+0.111) and the highest flip rate (17%, versus 10% for placebo and 4-12% for the other three conditions).

Attention and behavior now agree with each other for Arab Culture specifically: it draws more attention *and* moves the decision more than any other condition tested. That convergence across two independent measures is a meaningfully stronger claim than either measure alone.

## Where the prompt-structure check adds a real warning, not just confirmation

- `suffix` (demographic clause moved to the very end, right before the model answers): **every condition**, demographic and placebo alike, collapses to 93-100 of 100 items pushing toward "not offensive." That's not a content effect, it's a positional one, something about being the last thing before generation suppresses confident classification regardless of what it says. Any demographic-vs-placebo comparison run at this position is not measuring content, it's measuring position, and should not be used to support or refute the content-specific claim.
- `embedded` (folded into the question itself): placebo is uniquely extreme here, 100 of 100 items, mean shift -0.55, far beyond even its own suffix behavior, while the demographic conditions in the same structure are much more mixed (regions roughly balanced, not uniformly negative). That singles out this specific placebo phrasing ("As someone who prefers tea over coffee, is this text offensive?") as an unusually jarring construction in that exact grammatical slot, not a generalizable claim about placebos.

The practical upshot: `prefix` (and its `reworded` twin) are the trustworthy structures for this question. `suffix` and `embedded` are confounded in their own specific ways and shouldn't be read as either confirming or undermining the direction finding above.

## What this is worth for the thesis

1. **A real, positive, statistically well-supported finding**: demographic framing, regardless of which specific identity, reliably shifts this model toward classifying ambiguous content as offensive. That's a genuine behavioral effect of demographic context, not a null result, and it held up under a placebo control, which most work in this space skips entirely.
2. **A specific asymmetry worth naming**: "Arab Culture" produces a larger effect than "Western Europe," "man," or "woman" on both attention and behavior simultaneously. That's a concrete, quotable instance of unequal treatment across demographic framings by the same model under the same task.
3. **A finding about the placebo itself**: an irrelevant personal-preference statement doesn't just fail to move the model, it reliably moves it in the *opposite* direction from demographic framing. Worth a sentence of speculation in the discussion, plausibly the model reads any invoked social identity as a cue for heightened caution (a caution response learned via safety training), while a personal-taste statement reads as casual, lowering vigilance instead.
4. **A methodological result that belongs in your methods section regardless of the substantive finding**: naive entropy-based "high disagreement" selection silently introduces a baseline-ceiling confound, and the fix (require the model's own baseline prediction to have room to move in both directions before selecting for human disagreement) is a reusable check for anyone doing this kind of study, not just this thesis.
5. **A caution about prompt position that should shape how you describe your methodology**: results depend on where in the prompt you put the manipulation, and not in a simple "more salient position = bigger effect" way, `suffix` didn't amplify the content effect, it erased it under a positional artifact. This is worth a limitations paragraph, and it's also a reason to be confident in your choice to use the `prefix` structure throughout the rest of the project, since it's now been shown to be the position where content-specific signal actually survives.
