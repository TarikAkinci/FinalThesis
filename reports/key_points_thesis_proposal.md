# Key Points from Literature Review — Cross-Cultural Disagreement in Offensive Language Detection

## Core findings on why disagreement happens

1. **Demographics alone barely explain disagreement.** Across almost every paper (Hu & Collier; Kamruzzaman; Maurer), sociodemographic variables like age, gender, or race explain somewhere between 1–10% of variance in human ratings — while the text itself explains up to 70%. This is probably your strongest opening claim: single demographic labels are a weak proxy for "why people disagree."

2. **Demographics only matter in *interaction* with text features, not alone.** Maurer's paper (largest cross-dataset analysis, >25k items, >200k annotations) shows things like: younger annotators are desensitized to vulgarity but more sensitive elsewhere; older liberals and older conservatives diverge sharply on ideologically charged content. The pattern is never "group X rates higher," it's "group X rates higher *for this kind of content*."

3. **Intersectionality beats single-axis demographics.** GRASP (Prabhakaran et al.) found no significant group cohesion for isolated variables like gender or age alone, but strong, measurable cohesion for intersectional subgroups (Latina women, Black women, white men). Disagreement clusters around intersections, not categories.

4. **Moral values predict offensiveness better than nationality or demographics.** D3CODE and the related Davani (FAccT) paper both show that individual-level moral foundations — especially *Care* (harm avoidance) and *Purity* (degradation avoidance) — mediate cross-cultural differences in offensiveness ratings more than country or demographic averages do. A person's *individual deviation* from their country's moral average predicts their rating better than the country average itself.

5. **Cultural distance is measurable and predicts disagreement.** CreHate found pairwise annotator agreement is strongly negatively correlated (r = -0.658) with Kogut-Singh cultural distance — culturally close countries (UK/Australia) agree ~84% of the time, distant ones (Singapore/South Africa) ~74%. This gives you a quantifiable cultural variable to actually model against, rather than just categorical "country."

6. **A small, consistent set of linguistic triggers drives most cross-cultural disagreement:** sarcasm, personal bias on locally divisive topics, and differing tolerance for swear words show up again and again (CreHate: 31.7% / 27.3% / 10.0%) as the top three causes. D3CODE separately finds explicit mentions of social group identities (LGBTQ+, religious, ethnic groups) trigger the highest disagreement. This is a good candidate list for feature engineering or error analysis.

7. **Disagreement is not noise — it's a structural threshold conflict.** The HateXplain paper is a striking one: 73% of annotators who *disagree* on a label are actually highlighting the *same* words as justification (Jaccard similarity ≥ 0.3). People agree on what's relevant, they disagree on where the line is. This reframes the whole problem — you're not detecting annotator error, you're detecting where the boundary itself is contested.

8. **Disagreement concentrates at a specific fuzzy boundary, not randomly.** In HateXplain, 42.6% of all disagreement clusters specifically at the "hate speech vs. offensive speech" boundary, and offensive-labeled posts have a much higher disagreement rate (67.9%) than others. This suggests disagreement isn't spread evenly across the label space — it has a predictable "hot zone."

## Core findings on how LLMs behave (and fail) here

9. **LLMs are overconfident precisely where humans are uncertain.** The "overconfident judge" paper shows LLM accuracy crashes from ~87% on unanimous cases to under 65% on low-agreement ones, yet the models' internal confidence stays just as high. They don't get "less sure" the way humans do — this is probably your clearest motivating problem statement.

10. **This overconfidence has a systematic direction: LLMs over-censor.** Multiple papers converge on this — LLMs are biased toward labeling ambiguous/non-offensive content as offensive (accuracy on non-offensive disagreement cases drops to 45% in one paper), while separately being *too permissive* on things like sensitive advice. It's not random error, it's a consistent value skew.

11. **LLMs default to an artificial, flattened consensus.** Whether through persona prompting (Sarumi et al.: LLM-generated annotations show Krippendorff's alpha of 0.91 vs. 0.35 for real humans) or plain simulation (Movva et al.), models collapse the natural spread of human opinion into something far more uniform than reality. This is the "monolith trap" — models simulate the average member of a group, not its actual internal diversity.

12. **Persona/demographic prompting has a ceiling, and more isn't better.** Kamruzzaman et al. found alignment peaks at just 1–3 demographic attributes in a prompt and *degrades* past that ("over-specification"). Knowing which attributes matter for real human variance (via SHAP) gives zero predictive power over whether prompting with them helps the model — a nice cautionary point against naive "just add more demographic context" approaches.

13. **LLMs cannot pick up cultural context just by being told about it.** CreHate found that literally adding "is this offensive in Singapore?" to a prompt doesn't shift GPT-4's US-centric predictions at all. Cultural adaptation seems to require structural changes (culture tagging, multi-task learning) rather than prompt-level instruction.

14. **LLMs fail hardest at predicting *whose* disagreement will occur.** Movva et al. found LLMs can match the average human rating well, but have zero correlation with actual patterns of which demographic groups will disagree with each other. This is directly your thesis's core ask — models are decent at "is it offensive" and bad at "who will disagree and why," which is exactly the gap your project is targeting.

15. **Persona-prompted LLMs don't replicate real human bias patterns.** Giorgi's paper found 46% of human annotator-target demographic pairs show significant bias, versus only 3.3% for role-played LLMs — and the two bias patterns were essentially uncorrelated (r = -0.105). LLMs playing a persona don't reproduce that persona's actual sociological blind spots.

## Findings on modeling / methodological solutions

16. **Individual- and jury-level modeling beats aggregate labels.** Jury Learning (Gordon et al.) shows modeling individual annotators directly (MAE 0.61) beats both traditional aggregate models (MAE 0.90) and group-only models (MAE 0.81). It also lets you simulate custom "juries" (e.g., weighting marginalized groups more), which flipped final decisions on 13.6% of the most divisive items — a strong argument for pluralistic, non-single-label pipelines.

17. **No single prompting or modeling strategy wins everywhere — ensembling does.** Atil et al. found no persona-prompting method dominates across all demographics, but a simple learned ensembling approach (SVM over multiple prompt outputs) consistently beat every individual method and majority voting. This is a concrete, feasible architecture direction for your own modeling section.

18. **You can causally trace *which words* trigger an offensiveness judgment, and intervene surgically.** Feng et al.'s Token Distribution Dynamics method identifies the specific tokens driving an offensive classification well enough to neutralize just those tokens (rather than censoring whole posts), and Ghandeharioun et al. show that steering internal "who is asking" representations directly shifts a model's judgment of the same text — evidence that offensiveness is represented internally as a dynamic, perspective-dependent thing, not a fixed property. Useful if you want an interpretability angle alongside the annotation-modeling angle.

19. **Aggregation (majority vote) structurally discards minority signal, and you can't fix it downstream.** Multiple papers (Denton et al.; HateXplain) argue that once a majority vote collapses a subjective label, no amount of downstream model adjustment recovers the lost minority perspective — the fix has to happen upstream, at the data/label-representation stage (e.g., releasing unaggregated labels, graduated harm scales, explicit moral/value features).

## A few threads that repeat across almost every paper (good candidates for your intro/motivation)

- **"Disagreement is signal, not noise"** — stated explicitly or implicitly in nearly every paper, going back to Denton et al. (2021) as the conceptual root.
- **Single-label gold standards actively erase real, legitimate variation** rather than just being an approximation of it.
- **Item-specific / text-specific effects consistently outweigh person-specific effects** — but the two interact, and that interaction is where most of the unexplained variance lives.
- **Western/US/WEIRD bias is structural, in both datasets and LLMs**, and isn't fixed by surface-level prompt tweaks.
