# Is Demographic Attention Meaningful, or Just Token Count? Scaled Results (n=500)

Your supervisor asked three things before scaling up: log P(yes), automate and scale item selection, and add a placebo prefix to check whether the 5-10% demographic-attention figure means anything. All three are done, on 500 items (273 gender-disagreement, 227 region-disagreement, every one with a real human rater gap of at least 0.3), 2000 forward passes total. The run hit a bug partway through (see below, already fixed, no rerun needed), but the full dataset survived it and every number here comes from the corrected data.

## The two example items, and why these two

The script automatically picks the single highest-disagreement item per axis for the illustrated bar charts, not a hand-picked example. That's item 103 for region and item 3412 for gender.

**Item 103** ("I doubt anyone in Toronto has a brain, especially Simpson.", random category): Western Europe raters called this offensive 0% of the time (n=6), Arab Culture raters 87.5% of the time (n=8). The largest region gap in the whole 500-item set.

**Item 3412** ("Florence is to gay males what Yachats is to lesbians. Surely you know that.", social-group category): men rated this offensive 100% of the time (n=5), women 8.3% of the time (n=12). The largest gender gap in the whole set.

## Reading the two flowcharts

**Item 103.** The model already treats this as offensive with near-certainty at baseline (P(yes)=0.997), so nothing moves the final label anywhere, every condition still says "offensive." But the attention-share story is informative: "Arab Culture" pulls noticeably more attention (10.7%) than either "Western Europe" (8.1%) or the placebo (8.2%), and Western Europe and placebo land at almost exactly the same number. That's not an accident, it's this item acting out the aggregate pattern almost exactly: Arab Culture consistently beats the placebo, Western Europe does not reliably beat it at all.

**Item 3412.** Here the placebo actually pulls the most attention of the four conditions (8.2%), more than either gender_man (6.3%) or gender_woman (6.7%). That also matches the aggregate gender pattern. But look at the probability shift instead of the attention share: gender_man moves P(yes) by +0.018, gender_woman by +0.019, and the placebo only by +0.005. So for this one item, the condition that got *less* attention moved the decision *more*. That's a useful, concrete warning against reading "gets more attention" as "matters more." The two things came apart on this exact item, even though in the 500-item aggregate it's the placebo that wins on both measures for gender (more on that below). One item won't always match the population trend, which is exactly why 500 items and not 1 is what actually answers the question.

## The full 500-item result

This is the real headline, and it's not flattering for the demographic-attention story.

**Attention share, each condition vs. its axis's placebo, paired across items:**

| Condition | vs. placebo | Items where it's higher | p |
|---|---|---|---|
| gender_man | −0.018 (loses) | 0% of 273 | p = 1.6e-46 |
| gender_woman | −0.014 (loses) | 0% of 273 | p = 1.6e-46 |
| region_western_europe | −0.005 (loses) | 9% of 227 | p = 7.1e-33 |
| region_arab_culture | +0.021 (wins) | 100% of 227 | p = 5.4e-39 |

**Probability shift, |ΔP(yes)| vs. baseline, each condition vs. placebo:**

| Condition | vs. placebo | Items where it's higher | p |
|---|---|---|---|
| gender_man | −0.031 (loses) | 17% of 273 | p = 2.8e-15 |
| gender_woman | −0.023 (loses) | 19% of 273 | p = 6.5e-13 |
| region_western_europe | −0.027 (loses) | 26% of 227 | p = 3.1e-6 |
| region_arab_culture | +0.010 (wins, weakly) | 34% of 227 | p = 0.013 |

Every single comparison here is statistically decisive (n is large and the design is paired, so power isn't the issue). Reading it plainly:

- **A content-free placebo sentence beats "man", "woman", and "Western Europe" on both measures, every time, not a close call.** These three demographic conditions pull *less* attention and move the model's probability *less* than a sentence about preferring tea to coffee. Whatever that 5-9% figure from the earlier report was capturing, it wasn't really about demographic content for these three.
- **"Arab Culture" is the one real exception**, and even there the picture is mixed: it beats placebo on attention share cleanly and consistently (every one of 227 items), but its edge on actual probability movement is real yet weak, the mean favors it but only 34% of individual items do, and the p-value (0.013) is orders of magnitude less decisive than everywhere else in this table. That's a case where a skewed handful of large-effect items are pulling the average up, not a uniform per-item pattern the way the attention-share result is.

## Do we need to rerun anything before drawing a conclusion?

No. Here's why I'd stop here rather than send this back to the cluster:

- The sample is large (227-273 items per condition) and the effects are either enormous and consistent (p < 1e-12, 80-100% of items agreeing on direction) or, in the one exception's weaker half, still clears significance. More items would sharpen a decimal place, not change the conclusion.
- The dataset is clean: zero span-finding failures across all 2000 rows, confirmed directly (this run already uses the delimiter-based item-text lookup fixed after the item-1560 bug, so that failure mode can't recur here).
- The bug that crashed the original run (item_id not being a unique key across the two axes) is now fixed in the script and doesn't affect any number in this report, the repair reconstructed the correct 2000 rows from the same forward passes, it didn't need new ones.

What I would *not* claim is settled: this used one placebo phrase. It's a good one (your supervisor's own suggestion, plausible length-match), but a single wording is still a single data point about what "no demographic content" looks like. If you want to harden the placebo-specific conclusion further, a second, differently-worded placebo would be the next thing to add, not a rerun of this one. I don't think it's necessary to make the point you already have, just worth knowing it's there if a reviewer asks "what if the placebo mattered."

## Bottom line for the thesis

Adding demographic framing to a classification prompt does shift the model's attention and its output probability, that part of the earlier report holds. What this scaled, placebo-controlled run adds is a much less comfortable finding: for two of the three axes tested (man/woman, and Western Europe specifically), a completely unrelated sentence of similar length produces a *larger* shift than the real demographic phrase does. The one condition that shows a genuine, demographic-specific signal beyond generic prompt-perturbation is "Arab Culture", and even that signal is stronger on attention share than on the actual decision. That's a meaningfully different, more cautious story than "the model pays attention to demographic information", and it's the story the data actually supports.
