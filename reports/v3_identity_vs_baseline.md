# Is the model better without a demographic? And is there a group signal to align with?

Follow-up analyses on existing data, CPU only. Tables: `results/v3/analysis/q_identity_vs_unframed.csv`, plus the computations in this note.

## 1. Unframed vs framed vs steered, before and after the same calibration

Distance to the real group's average (mean |P(offensive) - share of that group's raters|), held-out texts (split B), regions and gender:

| | Llama prefix | Llama infix | Qwen prefix | Qwen infix |
|---|---|---|---|---|
| unframed, raw | 0.34 | 0.34 | 0.42 | 0.42 |
| framed, raw | 0.50 | 0.49 | 0.67 | 0.58 |
| unframed + constant shift | 0.19 | 0.20 | 0.20 | 0.20 |
| framed + constant shift | 0.19 | 0.19 | 0.24 | 0.23 |
| dampener-steered + constant shift | 0.17 | 0.19 | 0.22 | 0.22 |
| **all raters' share used as the prediction** | **0.10** | **0.10** | **0.10** | **0.10** |

The constant shift is fitted on one half of the texts and evaluated on the other.

- Raw, the unframed prompt is clearly closer to the real groups than the framed one.
- Calibrated, the identity adds nothing (Llama) or makes things worse (Qwen). Steering adds at most about 0.02.
- Simply predicting every group with the average of *all* raters is about twice as close as any model setting.

## 2. Do the groups in D3CODE differ, text by text?

For each group and text: the group's share minus everyone else's share. Raters of each text were split into two disjoint random halves, and the group gap was computed in each half. If groups react to specific texts in a stable way, the two halves should agree.

| group | texts | mean absolute gap | split-half agreement (Spearman) |
|---|---|---|---|
| Arab Culture | 1,041 | 0.147 | 0.00 |
| Indian Cultural Sphere | 1,141 | 0.148 | 0.01 |
| Latin America | 1,058 | 0.151 | 0.02 |
| North America | 1,014 | 0.138 | 0.05 |
| Oceania | 1,016 | 0.137 | 0.03 |
| Sinosphere | 1,242 | 0.161 | 0.10 |
| Sub-Saharan Africa | 1,039 | 0.163 | 0.11 |
| Western Europe | 1,039 | 0.135 | 0.01 |
| Man | 3,297 | 0.129 | 0.01 |
| Woman | 3,321 | 0.128 | 0.00 |
| *random pseudo-group, 1/8 of raters* | 1,562 | 0.157 | 0.02 |
| *random pseudo-group, 1/2 of raters* | 3,258 | 0.130 | 0.02 |

**Text-by-text group differences are almost entirely noise.** They are as large as those of randomly assembled groups, and they do not replicate between independent halves of the same group. The exceptions are a faint signal for Sinosphere and Sub-Saharan Africa. With 2 to 7 raters per region and text, this is partly a sample-size limit. It still means there is no reliable item-level group signal in this data for a model to follow.

## 3. Do the groups differ on average?

Share of ratings that call a text offensive, relative to the overall share:

- **Regions differ** (raters' individual shares across regions: p = 1e-37):
  - all D3CODE: Arab Culture +6.6 points, Latin America +5.6, Sub-Saharan Africa +2.3, Indian Cultural Sphere +1.9, Western Europe -2.0, North America -2.5, Sinosphere -5.6, Oceania -6.5;
  - our 500 texts: the same order, slightly compressed.
- **Gender does not** (men -0.5, women +0.6 points; p = 0.06).

## 4. Do the models get the regional ordering right?

Spearman correlation between the 8 real regional offsets (our 500 texts) and each model's region effect (prefix, length-adjusted, from v2):

- **Llama: 0.81** (n = 8, p ≈ 0.015). The ordering matches. But the regional spread is twice too large (0.21 in P vs 0.10 real), and every region sits about 0.2 above the unframed answer.
- **Qwen: 0.05.** No match.

This revises the v2 statement "region ranking rho 0.36, not significant" for Llama. That number used noisy per-text gaps. Against the stable average offsets, the ordering is clearly right.

## 5. Standard metrics, unframed prompt, all 4,519 D3CODE texts

Majority label: at least half of the raters call the text offensive (21.7% of texts).

| | Llama | Qwen |
|---|---|---|
| texts flagged | 78.7% | 85.0% |
| accuracy | 0.42 | 0.37 |
| precision | 0.27 | 0.25 |
| recall | 0.99 | 0.99 |
| F1 (offensive) | 0.43 | 0.40 |
| macro-F1 | 0.42 | 0.36 |
| ROC-AUC | 0.83 | 0.82 |
| Brier score vs raters' share | 0.26 | 0.35 |
| Spearman vs raters' share | 0.66 | 0.67 |

In standard terms: the models rank texts reasonably (AUC above 0.8) but their threshold is badly off. Recall is almost perfect and precision is low. That is the over-flagging in v2, stated in the usual metrics.

## 6. Moral-foundation groups and within-group disagreement

Same reliability test as in section 2, for raters split at the median of each moral foundation in D3CODE (rater-level scores in `d3-raters.csv`). The offset is the difference in each rater's share of "offensive" ratings, high vs low half (Mann-Whitney):

| split | texts | split-half agreement | average offset (points) | p |
|---|---|---|---|---|
| high care | 3,279 | 0.02 | +5.6 | 7e-18 |
| high equality | 3,221 | 0.05 | +2.1 | 8e-03 |
| high proportionality | 3,146 | 0.00 | +2.6 | 2e-04 |
| high authority | 3,141 | 0.06 | +3.8 | 2e-08 |
| high loyalty | 3,092 | 0.01 | +3.7 | 3e-07 |
| high purity | 3,341 | 0.07 | +2.9 | 8e-05 |
| woman vs rest | 3,321 | 0.00 | +1.3 | 0.05 |
| random half (null) | 3,258 | 0.02 | -0.1 | 0.9 |

- Groups defined by moral values behave like the demographic groups. They differ by a small average strictness (high-care raters are the strictest), but not text by text.
- **Disagreement is mostly within groups.** Mean entropy of the binary ratings among raters of the same region on a text is 0.64, against 0.80 among all raters of that text (1.0 = maximal). Most of the disagreement remains inside a region.
- Conclusion for the thesis: no group attribute available in D3CODE (region, gender, moral foundation) predicts how individuals judge specific texts. Only small average offsets exist. A moral-values framing experiment would most likely repeat the demographic result, so it was not run.
