# v2 behavioural analysis (Phase 2)

Data: the v2 grid on COMA (H200), Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct. Both models saw the same 500 items, under the same 22 conditions (14 demographics, 8 content-free placebos) and the same 10 prompt variants. That is 110,000 framed prompts plus 500 baselines per model, with no missing values.

Scripts: `analysis_v2_behaviour.py` and `analysis_group_gap.py`. All tables are in `results/v2/analysis/behaviour_*.csv` and `group_gap_*.csv`.

## 0. Is the data trustworthy?

- Row counts, spans and splits are all correct. Every (variant, condition) has exactly 500 items. The identity span has the same token length on every item and in both models.
- Numerical noise is zero. The same prompt run twice gives the same answer to the last digit (4,420 repeats per model). Grid baselines match the scoring run to within 1e-16.
- Reproduction against the LRZ (v1) run: rankings agree almost perfectly (Spearman 0.995 or higher) and there is no systematic shift. Single items differ by about 0.1 in log-odds (median). Part of that is the bf16 rounding of the output scores in v1, which v2 removes (v1 had only 58 distinct P values for 236 Llama items). The rest is ordinary bf16 differences between GPUs and library versions. So v2 replaces v1, and the two should not be mixed item by item.

## 1. How to read the numbers

- The main measure is the shift in log-odds of "yes" compared with the item's own unframed baseline (`dlo`). A shift of +1 means the odds of "offensive" are multiplied by e (about 2.7). Log-odds do not get squeezed near 0 and 1, so items with different baselines can be compared. Average probabilities are given where they help.
- Every phrase is identical on all 500 items. A test that uses items as the unit therefore mostly asks "do these tokens differ from those tokens", and it comes out significant for almost anything. Claims about identity content use the phrase as the unit instead:
  - demographics against placebos, comparing condition means. In this fully crossed design that is exactly the fixed-effect test of the mixed model `shift ~ is_demographic + (1|item) + (1|phrase)`;
  - with phrase length as a covariate (see section 3 for why);
  - for a single demographic: where it falls compared with a new content-free phrase of the same length (a prediction interval from the 8 placebos).
- Variance components: in every variant, the variance between phrases has the same order of magnitude as the variance between items. Which phrase is inserted matters about as much as which text is judged.

## 2. Headline: which identities do something that content-free phrases do not?

Demographics against placebos, comparing condition means, **adjusted for phrase length**. The table shows the difference in log-odds and the p-value with the phrase as the unit, for the prefix variant:

| axis | Llama diff | Llama p | Qwen diff | Qwen p |
|---|---|---|---|---|
| all 14 demographics | +1.21 | <0.001 | +2.38 | <0.001 |
| region (8) | **+1.62** | **<0.001** | **+2.78** | **0.001** |
| gender (2) | +0.22 | 0.59 | +0.27 | 0.79 |
| religion (2) | -0.10 | 0.80 | +1.17 | 0.33 |
| bare identity, "an Arab" / "a European" (2) | +0.70 | 0.13 | +0.96 | 0.34 |

- **Region is the one axis that clearly carries content.** It differs from content-free phrases of the same length in every one of the 10 variants, in both models (largest p: 0.003 for Llama, 0.006 for Qwen). Region framings push the model toward "offensive" more than any content-free phrase of that length does.
- **Gender, religion and the bare identity words do not.** "You are a woman", "You are a Muslim" and "You are an Arab" behave like "You are a runner" in the prefix family of both models. There are specific exceptions, listed in section 5.
- With only 2 phrases per axis, gender, religion and identity have little power at the phrase level. "Not distinguishable from placebos" is the honest wording, not "no effect". Still, their point estimates sit inside the placebo spread, not just below a significance threshold. Llama's gender estimate, for example, is +0.2 log-odds, where placebos of the same length range from +0.6 to +1.0.
- In probabilities (averages over items; the baseline average is 0.47 for Llama and 0.53 for Qwen): under prefix, region framings give P(offensive) 0.70 in Llama and 0.87 in Qwen. Placebos give 0.47 and 0.70.

Single regions under prefix, against a content-free phrase of the same length (log-odds; * means p < 0.05 at the phrase level):

| region | Llama | Qwen |
|---|---|---|
| Sub-Saharan Africa | +2.97* | +5.56* |
| Arab Culture | +1.99* | +3.98* |
| Sinosphere | +1.36* | +3.92* |
| Indian Cultural Sphere | +1.51* | +3.68* |
| Latin America | +2.17* | +1.26 |
| Western Europe | +1.27* | +2.29 |
| Oceania | +1.18* | +2.34 |
| North America | +0.98 | +1.71 |

- Sub-Saharan Africa is the strongest region in both models, which confirms the v1 finding.
- Even Western Europe and North America push upward. The region effect is not "non-Western only", although in Qwen the non-Western regions are clearly stronger.

## 3. The placebos: what content-free phrases do

- **Longer content-free phrases shift the answer less, or downward.** Across the 8 placebos, the Spearman correlation between length and shift is negative in every variant of both models (-0.29 to -0.93). Demographic phrases do not follow that pattern. Region phrases are the longest (7 to 9 tokens), so comparing them with the average placebo would understate their effect, and comparing gender with long placebos would overstate it. That is why all claims above are length-adjusted.
- **In Qwen almost any persona pushes toward "offensive".** Under prefix, 6 of the 8 placebos raise the log-odds by +2.3 to +3.8 ("You are a cyclist." takes Qwen from 0.53 to 0.85 on average). Demographics add another +2.4 on top. In Llama, placebos sit around zero (+0.03 on average).
- **The tea sentence is an outlier**, as in v1. In Llama it lowers the log-odds by -5.2 (embedded) and -7.5 (target). A question about whether a text is offensive "for someone who prefers tea over coffee" is answered "no" almost always. It is kept in the placebo set (removing it does not change any conclusion; both versions are in `behaviour_identity_specific.csv`).
- Comparing each demographic only with its 2 exact-length placebos is sensitive to which 2 they are. The four long regions share "stairs" and "tea", the two lowest placebos, so their raw contrasts look larger than they are. The length line through all 8 placebos is the more stable reference, and it is what sections 2 and 5 use.

## 4. Robustness to wording

- The four identity templates ("You are X." / "Assume you are X." / "Imagine you are X." / "Consider that you are X.") agree almost perfectly. Over the 22 conditions, the condition means correlate at Spearman 0.98 to 0.99 for Llama and 0.90 to 0.93 for Qwen. The identity-specific effects correlate at 0.91 to 0.99 and 0.83 to 0.95.
- So the effects come from *who* is named, not from how the sentence is built. This was the main reason for adding paraphrases, and it holds.

## 5. Prompt structure

Mean P(offensive) for demographics vs placebos per variant (baseline average 0.47 for Llama, 0.53 for Qwen):

| variant | Llama demo | Llama placebo | Qwen demo | Qwen placebo |
|---|---|---|---|---|
| prefix | 0.66 | 0.47 | 0.87 | 0.70 |
| infix | 0.64 | 0.39 | 0.76 | 0.58 |
| suffix | 0.24 | 0.17 | 0.22 | 0.53 |
| embedded ("As X, is ...") | 0.50 | 0.27 | 0.65 | 0.50 |
| target ("offensive for X?") | 0.50 | 0.16 | 0.51 | 0.15 |
| context | 0.74 | 0.54 | 0.75 | 0.65 |
| context_long | 0.67 | 0.49 | 0.84 | 0.77 |

- **Position.** Prefix and infix behave alike: an identity before the question pushes upward, and region carries content in both. **Suffix pulls everything down** in both models, because the last thing the model reads is no longer the question. In Qwen the suffix even **reverses the identity effect**: demographics push down much more than placebos (-2.2 log-odds after length adjustment, p < 0.001, and every axis is significant there). So in Qwen, where the identity sits decides the direction of its effect.
- **Embedded vs target.** These are different questions. "As a woman, is this offensive?" asks for a perspective. "Is this offensive for a woman?" asks whether the text targets that group. Under target, placebos collapse ("offensive for a runner?" → almost always no), so the gap between demographics and placebos is the largest of all variants (Llama +3.3, Qwen +5.0 length-adjusted). This measures topic relevance, not perspective taking. It should be reported separately and not mixed with the perspective variants.
- **Context.** The elaboration sentence ("Your background and everyday experience shape how you read things.") is now identical for demographics and placebos. It does not make region stronger. In Llama it does give gender and the bare identity words an effect: length-adjusted gender +1.07 (p = 0.007) under context and +1.07 (p = 0.031) under context_long, where gender was +0.22 (p = 0.59) under prefix. "Muslim" also becomes positive in Llama only under the context variants (+0.95, p < 0.05). In Qwen, context does not add anything for gender or religion. So telling Llama that background matters is what makes it use gender; the bare phrase does not.

## 6. Confidence and flips

- The items were chosen to be borderline, so flips are common. Under prefix, 29% of items flip for demographics against 21% for placebos (Llama), and 38% against 34% (Qwen).
- Confidence, measured as distance from 0.5: in Qwen, demographics raise confidence more than placebos (+0.18 vs +0.10 under prefix). In Llama, confidence barely changes (+0.02 vs -0.03). The v1 observation that placebos make Llama less sure is not visible in v2. It was most likely specific to the old placebo set.

## 7. What predicts how strongly an item reacts?

Item-level identity-specific effect (demographics minus placebos, averaged over the 4 templates):

- **Human disagreement does not.** Spearman with rating entropy is 0.03 (p = 0.46) in Llama and 0.08 (p = 0.06) in Qwen. Contested items are not more sensitive to identity framing.
- **The human offensive rate does not** either (0.01 and 0.03).
- **Category matters a little in Qwen.** Social-group items show a smaller identity-specific effect (2.2 vs 3.0 log-odds, Kruskal p = 0.019), although any phrase moves them more (4.5 vs 3.9). Llama shows no clear difference (p = 0.067).
- **The baseline matters for the general shift, not for the identity part.** In Llama, items with a high baseline drop and items with a low baseline rise under any phrase (Spearman -0.63). That is a pull toward the middle that every phrase shares. The identity-specific part is only weakly related (-0.16).

## 8. Does framing bring the models closer to the real group?

No, in every way we can measure it.

- **Absolute distance.** Mean absolute difference between the model's P(offensive) and the share of that group's raters who called the item offensive, under prefix (averaged over gender and the 8 regions):
  - Llama: 0.35 unframed, 0.51 framed, 0.34 with the matched placebo;
  - Qwen: 0.40 unframed, 0.68 framed, 0.50 with the matched placebo.

  Framing moves the models *away* from the real groups, and further away than a content-free phrase does. The real rates are low (about 0.13 to 0.26), and framing pushes upward. The only variant that reduces the distance is suffix (Llama 0.22, Qwen 0.21), and only because it pushes predictions down toward the low real rates. In Llama the placebos get even closer there (0.18). In Qwen the demographics beat the placebos (0.36) under suffix, but only because of the reversal in section 5, which pushes demographics down hardest. Nothing about this is specific to the named group.
- **Relative gaps across items.** Do items where a group is more offended than everyone else get a larger push under that group's framing? 0 of 240 tests are significant after correction (all variants, both models). Mean Spearman is 0.02 for Llama and 0.00 for Qwen.
- **Ordering of regions within an item.** Across items, Llama orders the regions a little like the raters do (mean rho 0.10, p = 0.0002). After removing each region's average effect, nothing is left (0.015, p = 0.57). So this comes from a fixed ranking of regions, not from anything item-specific. For Qwen the value is 0.
- **Region ranking overall.** The correlation between the models' 8 region effects and the 8 regions' average rater gaps is 0.36 for Llama (n = 8, p = 0.39) and 0.00 for Qwen. The v1 values (+0.62 and +0.67) do not hold up with the larger and matched v2 data.

This is the central motivation for the steering stage. The models clearly use region information, but not in a way that resembles the real groups.

## 9. Llama vs Qwen

- Condition means correlate at Spearman 0.67 across all 220 (variant, condition) cells. Under target the agreement is high (0.88), because both models treat "offensive for X" as a relevance question. Under suffix it is negative (-0.34), because Qwen reverses there and Llama does not.
- Identity-specific effects agree only moderately (0.47 overall; 0.50 under prefix). Both models single out Sub-Saharan Africa, but their gender and religion patterns differ. In Qwen under prefix, Muslim > Christian (+2.4 vs 0.0); in Llama both are at about -0.1.
- Qwen's effects are 2 to 3 times larger in log-odds, and it reacts to almost any persona.

## 10. What changed compared with v1

| v1 statement | v2 |
|---|---|
| Real framings beat the placebo pool on about 93% of items | Holds at the phrase level for all demographics together, but it is carried by region. Gender, religion and the bare identity words are not distinguishable from same-length placebos. |
| Sub-Saharan Africa is the top region in both models | Confirmed |
| Muslim has no effect in Llama | Confirmed, and the same holds for Christian. Only the context variants give Muslim an effect. |
| Placebos reduce confidence in Llama | Not seen in v2 |
| Group level, the model's region ranking correlates +0.62 / +0.67 with the raters' gaps | Not confirmed: 0.36 (n.s.) and 0.00 |
| Framing increases the distance to the real group | Confirmed and larger: 0.35 to 0.51 (Llama), 0.40 to 0.68 (Qwen) |
| Suffix collapses everything; Qwen reverses there | Confirmed with exact controls |

## 11. What this means for the next steps

- **Steering targets.** Region is the only axis with a content effect that exists in both models and in every prompt structure. That makes the region conditions the natural targets for the per-head analysis and for steering. Gender and religion are more interesting as cases where steering might *create* an effect that prompting alone does not produce (with Llama's context variant as the prompt-only comparison).
- **Structures.** The prefix family (prefix plus the 3 paraphrases) and infix give a consistent effect. Suffix behaves qualitatively differently in Qwen. Target measures something else. For the head analysis I'd use the prefix family as the main set and infix as the replication.
- **Success criterion for steering.** No condition moves the models toward the real groups now. A steering result that reduces the distance to the group, or that creates item-level tracking of the group gaps on held-out items, would be a genuinely new effect. Amplifying the existing effect would most likely increase the distance, because the existing effect points away from the groups.

## 12. Caveats

- The items are borderline for both models: milder (human offensive rate 0.21, against 0.34 in all of D3CODE) and less contested than D3CODE overall. The results describe how framing works on texts the models are unsure about.
- 2 phrases per axis for gender, religion and identity give little power at the phrase level. The paraphrases help with wording, not with the number of distinct identities.
- Region is also the axis with the most distinct phrases (8). Its clearer result is partly a matter of power. The size of the region effects (+1 to +3 log-odds in Llama, +1.3 to +5.6 in Qwen) makes it unlikely that power is the whole story.
- The answer is a single token, 0 or 1. Nothing here says how the models would explain or grade their judgments.
- Religion cannot be checked against real raters, because D3CODE has no rater religion.
