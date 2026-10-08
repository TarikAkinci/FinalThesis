# Master table: full analysis and interpretation

Data: the completed master-grid run from 25 Sept (`results/master_grid/master_grid_*.csv`, summarized in `results/master_table.csv`). The run currently on the cluster adds the `infix` and `target` structures and the prefix/infix/suffix charts. Everything below should still hold once that lands. Section 15 lists what to check when it does.

All numbers were computed directly from the raw per-item rows, not copied from earlier reports.

---

## 0. What the data actually is

- **Models:** Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct, bf16, one deterministic forward pass per prompt.
- **Items:** the seeded 500-item D3CODE evaluation sample. Every item gets a baseline (no framing). Only items whose baseline P(yes) is between 0.02 and 0.98 go through the condition grid: **236 items for Llama, 64 for Qwen**. Qwen has far fewer because it is pinned at 0 or 1 on most items (396 of 500 above 0.98).
- **Conditions (21):** 8 D3CODE regions, man and woman, Christian and Muslim, bare identity ("You are Arab." and "You are European."), and 7 placebo phrases (6 distinct; `placebo_tok9` and `placebo_long` are the same tea sentence).
- **Structures:** prefix (all 21 conditions), reworded, embedded and suffix (5 core conditions plus the token-matched pairs), and context and context_long (6 conditions, known to be confounded, see 7.5).
- **Main outcome:** signed change in P(yes) against the same item's baseline. Other measures: flip rate, change in prediction entropy, and sink-excluded attention shares.

A note on p-values: per-item Wilcoxon tests in this project are very small (often below 1e-10) because they treat 236 items as independent, while the phrase is identical on all of them. They show that an effect is consistent across items. They overstate how certain we are about the phrase itself. I report them where useful but lean on effect sizes and item counts.

---

## 1. Baseline behaviour: the models over-flag, and they are most sure where humans disagree most

### 1.1 Massive over-prediction of "offensive"

| | Llama | Qwen |
|---|---|---|
| Mean P(yes), all 500 items | 0.768 | 0.851 |
| Predicted offensive | 78.8% | 85.4% |
| Human majority says offensive | 20.0% | 20.0% |
| Mean fraction of raters saying offensive | 0.331 | 0.331 |
| Agreement with the human majority label | 40.8% | 34.6% |
| Called offensive when most humans said not offensive | 73.8% | 81.8% |

Both models call about four times as many items offensive as human raters do. This is the over-censorship pattern from the literature (Lu et al.), and it is very pronounced here. It also sets the scene for everything below: the model starts far above what any human group thinks, so a push toward "offensive" moves it further from people, not closer.

### 1.2 The models rank items sensibly but are miscalibrated

- Spearman between P(yes) and the human fraction offensive: **0.64 (Llama), 0.56 (Qwen)**. The ordering of items is reasonable.
- The level is wrong, and so is the confidence profile:

| Human agreement tercile | Llama confidence | Qwen confidence | Human fraction offensive |
|---|---|---|---|
| Most contested | 0.911 | 0.988 | 0.469 |
| Middle | 0.814 | 0.962 | 0.365 |
| Most agreed | 0.748 | 0.922 | 0.160 |

Confidence here is |P(yes) − 0.5| × 2. Both models are **most confident on the items humans disagree about most**. Correlation between model confidence and human agreement: **−0.32 (Llama), −0.47 (Qwen)**. Model uncertainty and human rating entropy are also negatively related (−0.37, −0.46). This is the opposite of what a calibrated model would do.

A caveat: the contested items are also the ones with the most offensive-looking content (human fraction offensive 0.47 against 0.16), so "contested" and "content that looks offensive" overlap. Part of this pattern is the models confidently flagging edgy text that splits humans.

### 1.3 By D3CODE category

| Category | n | Llama P(yes) | Qwen P(yes) | Humans |
|---|---|---|---|---|
| social-group | 267 | 0.866 | 0.947 | 0.409 |
| random | 190 | 0.699 | 0.797 | 0.262 |
| moral | 43 | 0.465 | 0.489 | 0.152 |

Text mentioning social groups is where over-flagging is strongest (Qwen calls 95% of it offensive while 41% of raters do). This matches D3CODE's own finding that explicit group mentions drive disagreement.

---

## 2. Where attention goes

Baseline, in-band items:

| | Llama | Qwen |
|---|---|---|
| Attention sink (template tokens) | 0.798 | 0.662 |
| Instruction share, sink excluded | 0.867 | 0.847 |
| Item text share, sink excluded | 0.133 | 0.153 |

- **Most attention at the decision token goes to fixed template tokens**, about 80% in Llama and 66% in Qwen. This is the known attention-sink effect (Xiao et al.). It replicates in a second model, with a smaller but still dominant sink.
- **Of the content attention, the instruction gets roughly 85% and the text being judged only about 13-15%.** The model attends far more to "is this offensive, answer 0 or 1" than to the text it is judging.
- **A demographic phrase takes its share from the instruction, not from the item text.** In prefix, instruction share drops from 0.867 to about 0.76-0.82 depending on the phrase, while item text stays at 0.13-0.15. This holds in both models.
- The demographic phrase itself gets only 4-10% of content attention in prefix. That is small, yet it moves predictions a lot (section 3). Section 9 shows that this attention share mostly tracks phrase length.

---

## 3. The core effect: real framings move predictions toward "offensive" beyond what placebos do

### 3.1 Against the full placebo pool

For each item, I compared the mean shift over all 14 real framings with the mean shift over the 6 distinct placebo phrases (prefix):

| | Llama | Qwen |
|---|---|---|
| Placebo pool mean shift | **−0.023** | **+0.150** |
| Real framings minus placebo pool | **+0.131** | **+0.238** |
| Items where real > placebo | 94.5% | 92.2% |

This is the cleanest single result in the project. It doesn't depend on any one placebo sentence, it holds in both models, and it holds on more than 9 in 10 items.

### 3.2 Against exact token-matched placebos

Real phrase minus a placebo of identical token length, same items. A star means p < 0.05 (paired Wilcoxon).

| Structure | Model | Muslim (4 tok) | Woman (5) | W. Europe (7) | Arab Culture (9) |
|---|---|---|---|---|---|
| prefix | Llama | +0.017 | +0.110* | +0.142* | +0.253* |
| prefix | Qwen | +0.063* | +0.167* | +0.104* | +0.458* |
| reworded | Llama | +0.018 | +0.195* | +0.196* | +0.264* |
| reworded | Qwen | +0.195* | +0.264* | +0.229* | +0.561* |

Length doesn't explain the effect. In the two trusted structures, every comparison is positive, and all are significant except Muslim in Llama.

### 3.3 Signed versus unsigned

Mean absolute shift is similar for real phrases and placebos (Llama prefix: placebos 0.10-0.16, real 0.12-0.26). Looking only at magnitude suggests the phrase is noise. The signed shift shows the real phrases push consistently in one direction while placebos scatter. This methodological point stands, and section 5 refines how to phrase the placebo side.

### 3.4 Flips go mostly one way

- Llama prefix: 836 flips, **71% from "not offensive" to "offensive"**.
- Qwen prefix: 493 flips, **87% toward "offensive"**.

---

## 4. Breadth: which demographics move the model most

### 4.1 Llama, prefix, ranked by signed shift

| Condition | Tokens | Shift | Pushed up | Flip | Attention |
|---|---|---|---|---|---|
| Sub-Saharan Africa | 9 | **+0.246** | 87% | 28% | 0.085 |
| Latin America | 7 | +0.216 | 85% | 27% | 0.086 |
| Arab Culture | 9 | +0.144 | 73% | 24% | **0.097** |
| Western Europe | 7 | +0.135 | 78% | 17% | 0.070 |
| "You are European." | 4 | +0.103 | 73% | 14% | 0.058 |
| Oceania | 8 | +0.102 | 68% | 17% | 0.071 |
| North America | 7 | +0.102 | 72% | 14% | 0.067 |
| Indian Cultural Sphere | 9 | +0.085 | 65% | 18% | 0.093 |
| "You are Arab." | 4 | +0.085 | 69% | 17% | 0.064 |
| man | 5 | +0.078 | 71% | 12% | 0.059 |
| Sinosphere | 9 | +0.077 | 55% | 17% | 0.083 |
| Christian | 4 | +0.075 | 70% | 11% | 0.059 |
| woman | 5 | +0.059 | 60% | 12% | 0.063 |
| **Muslim** | 4 | **+0.006** | **49%** | 19% | 0.074 |

### 4.2 Qwen, prefix

Every real condition shifts by +0.35 to +0.46, with 92-100% of items pushed up. Sub-Saharan Africa is again the largest (+0.462). The spread between conditions is narrow, so Qwen responds to "any identity" much more uniformly than Llama.

### 4.3 Findings

1. **Arab Culture is not special.** In Llama, Sub-Saharan Africa and Latin America shift more. In Qwen, Arab Culture sits mid-pack. The earlier Arab-Culture-versus-Western-Europe result was a two-region comparison that couldn't show this. What Arab Culture does have is the **highest attention share** in both models (0.097 and 0.072), so it is special in attention but not in behaviour.
2. **Sub-Saharan Africa is the top region in both models.** Cross-model agreement on the 8-region ranking is moderate (Spearman 0.60). Over all 14 real conditions it is weaker (0.35), mostly because the Qwen spread is so narrow.
3. **Muslim is the exception in Llama.** Shift +0.006, items split 49/51, and not different from its matched placebo (p≈0.25). It turns negative in reworded (−0.040) and strongly negative in embedded (−0.291) and suffix (−0.419). Christian, at the same length, behaves like every other framing (+0.075). In Qwen, Muslim behaves normally (+0.366). One possible reading is a safety-tuning behaviour in Llama around Muslim identity: the model avoids treating text as more offensive from a Muslim perspective, or a Muslim cue triggers a "don't stereotype" response. I can't test that with this data, so it is a hypothesis to state with care. It is still a striking, model-specific asymmetry worth reporting.
4. **The short and long forms of an identity behave differently.** "You are Arab." gives +0.085 and "someone from the Arab Culture region" gives +0.144. "You are European." gives +0.103 and "someone from Western Europe" gives +0.135. The longer region phrases shift a bit more. Combined with section 9 (longer means more attention), wording and length matter alongside the identity itself.
5. **Gender has the smallest effect** among real framings in Llama (man +0.078, woman +0.059), and woman is the least consistent (60% pushed up).

---

## 5. The placebos are not one thing

Prefix shift by placebo phrase:

| Placebo | Tokens | Llama | Qwen |
|---|---|---|---|
| "You are a swimmer." (mid) | 6 | **+0.044** | **+0.375** |
| "You are tall." (tok4) | 4 | −0.011 | +0.303 |
| "You are an early riser." (tok7) | 7 | −0.008 | +0.282 |
| "You are punctual." (tok5) | 5 | −0.051 | +0.199 |
| "You are left-handed." (short) | 5 | −0.004 | −0.183 |
| tea over coffee (long, tok9) | 9 | **−0.108** | −0.074 |

- **The tea sentence, used in every earlier study, is the most negative placebo in Llama.** It is an outlier, not a typical control. The earlier headline "the placebo pushes the model the other way" was mostly a property of that one sentence.
- **In Qwen, most placebos push toward offensive too**, sometimes nearly as much as real framings (swimmer +0.375 against real +0.35 to +0.46). For Qwen, adding any "You are X." sentence makes it more likely to flag. The real identity adds about +0.24 on top of that (section 3.1).
- **In Llama, the placebo pool averages close to zero (−0.023).** So in Llama, a neutral persona does almost nothing on average, and a real identity adds +0.13.
- This changes how to phrase the finding. "Placebos push the other way" doesn't hold up. "**Real identities push toward offensive beyond what content-free personas do, and content-free personas have phrase-dependent effects of their own**" does.

---

## 6. Confidence: real framings make the model surer, placebos less so (in Llama)

Change in prediction entropy (negative means more confident), prefix:

| | Llama | Qwen |
|---|---|---|
| Real framings (mean) | −0.018 | **−0.240** |
| Placebo pool | **+0.084** | −0.118 |

- **In Llama, real identities leave confidence roughly unchanged or raise it**, with the strongest regions making it clearly surer (Sub-Saharan Africa −0.188, Latin America −0.136). **Placebos make it less sure** (+0.084 on average, up to +0.165 for left-handed).
- **In Qwen, everything increases confidence**, and real identities do so about twice as much as placebos.
- This direction (identity makes the model surer that text is offensive) adds to the over-confidence problem from section 1. Framing doesn't open up uncertainty. If anything it closes it.

---

## 7. Prompt structure

Mean shift for real framings and placebos, and the gap between them:

| Structure | Llama real | Llama placebo | Llama gap | Qwen real | Qwen placebo | Qwen gap |
|---|---|---|---|---|---|---|
| prefix | +0.108 | −0.035 | +0.143 | +0.388 | +0.118 | +0.270 |
| reworded | +0.053 | −0.120 | +0.173 | +0.388 | +0.043 | +0.345 |
| embedded | −0.106 | −0.259 | +0.153 | +0.139 | −0.073 | +0.212 |
| suffix | **−0.289** | **−0.324** | +0.035 | **−0.268** | −0.049 | **−0.218** |
| context | +0.258 | −0.134 | +0.392 | +0.493 | +0.374 | +0.118 |
| context_long | +0.291 | −0.208 | +0.499 | +0.520 | +0.241 | +0.279 |

### 7.1 Prefix and reworded agree
These are the two clean structures. Reworded gives the largest real-placebo gap in both models. Effects there are robust to a change of wording.

### 7.2 Suffix shifts everything downward, in both models
Llama: 3% of real-framing rows go up. Suffix also has the highest flip rate (38%). The phrase gets about twice as much attention as in prefix (0.158 against 0.073) because it sits right before the answer position, which is the recency effect (Zhao et al.). Two model differences:
- In Llama, real and placebo collapse together (gap only +0.035). Content barely matters.
- **In Qwen, real framings push down harder than placebos** (gap −0.218). In suffix position, Qwen treats a real identity as a reason to call the text *less* offensive. That is the reverse of prefix. So structure doesn't only add noise. It can flip the direction of the identity effect.

### 7.3 Embedded is model-dependent
Llama shifts down overall, but the gap to placebo survives (+0.153). Qwen stays positive. The extreme case is Llama's tea placebo at −0.520, with all 236 items pushed down. Token-matched pairs in embedded are mixed in sign (woman −0.170 and Muslim −0.032, but Arab Culture +0.508).

### 7.4 Matched pairs by structure (real minus same-length placebo)
In suffix, Llama's gender and Muslim pairs reverse sign (woman −0.176, Muslim −0.113), while the region pairs stay positive. In Qwen, all four reverse. The real effect is stable across prefix and reworded, but position can break it.

### 7.5 Context blocks: large numbers, known confound
Context effects are the largest in the table (Llama gap +0.39 to +0.50). But the demographic elaboration contains "...what you find **offensive**" and the placebo elaboration contains "...what you find **refreshing**". The task word only appears in the real version, so it isn't a clean test of "more context". Treat this as a lead and don't cite the size. The fix is to give the placebo the same "offensive" sentence.

---

## 8. Headroom: the baseline predicts how much an item moves

This is an important methodological finding and it refines the "uncertain items shift twice as much" result from before.

**Baseline P(yes) predicts the shift very strongly**: Spearman −0.71 (Llama) and −0.77 (Qwen), and −0.84 and −0.78 for the absolute shift. Items the model already calls offensive barely move. Items it calls not offensive move a lot.

Llama, mean over real framings, by baseline P(yes):

| Baseline P(yes) | n | Real shift | Placebo pool shift | Gap |
|---|---|---|---|---|
| 0.00-0.20 | 36 | +0.324 | +0.111 | +0.213 |
| 0.20-0.35 | 22 | +0.252 | +0.065 | +0.187 |
| 0.35-0.65 | 39 | +0.189 | +0.005 | +0.184 |
| 0.65-0.80 | 30 | +0.086 | −0.069 | +0.155 |
| 0.80-1.00 | 109 | −0.015 | −0.082 | +0.067 |

What this shows:
1. **The earlier "uncertain items shift about twice as much" is really a headroom effect.** The shift is largest for low-baseline items, not specifically near 0.5.
2. **Placebos follow the same slope**: up at low baselines, down at high ones. Any perturbation pulls predictions toward the middle. This is a regression-to-the-middle effect common to all inserted sentences.
3. **The real-minus-placebo gap is positive in every bin, on 93-100% of items.** So the identity effect isn't explained by headroom. It is smallest at the ceiling (+0.067), where there is little room left. Qwen shows the same positive gap in every bin with enough items.
4. **Consequence for analysis:** compare real framings against placebos on the same items, or control for baseline P(yes). Raw shifts are confounded by where an item starts. This is also why the baseline band filter was needed, and why Qwen, with mostly extreme baselines, keeps only 64 items.

---

## 9. Attention share mostly measures phrase length, not influence

### 9.1 Across conditions (prefix)

| Correlation | Llama | Qwen |
|---|---|---|
| Phrase tokens vs attention share | **+0.72** | **+0.90** |
| Attention share vs shift (all 21 conditions) | +0.44 | +0.20 |
| Attention vs shift, real conditions only | +0.42 | −0.02 |
| Attention vs shift, placebos only | −0.64 | −0.35 |

### 9.2 Within a condition, across items
The mean item-level correlation between attention to the phrase and the shift is **+0.14 (Llama) and +0.04 (Qwen)**.

### 9.3 Interpretation
- **Head-averaged attention share is mostly a function of how many tokens the phrase has.** Among placebos, more attention even goes with a *more negative* shift, which is mainly the long tea sentence again.
- Arab Culture gets the most attention and isn't the strongest behaviourally. Sub-Saharan Africa gets average attention and is the strongest.
- **The amount of attention a phrase gets, averaged over all layers and heads, is a weak predictor of what the phrase does to the decision.** This is not proof that attention is irrelevant: averaging over 1,024 heads (Llama) or 784 (Qwen) can wash out a few heads that matter. It does mean the head-averaged numbers used so far can't identify the mechanism.
- **This is the strongest argument for doing per-head analysis before steering.** If the effect lives in a few heads, it is invisible in the averages, and steering on averaged attention would target the wrong thing. The per-head arrays from this run (`master_grid_perhead_*.npz`) are exactly what is needed, and haven't been analysed yet.

---

## 10. Does framing make the model behave like the real group?

This is the question steering is ultimately about. Three levels:

### 10.1 Item level: no
For each item, the gap between a group's ratings and everyone else's, against the shift under that group's framing (8 regions plus gender, both models): **1 of 20 tests significant, mean correlation 0.00.** Framing does not make the model sensitive to the items that particular group finds more or less offensive.

### 10.2 Group level: a weak hint
Ranking the 8 regions by how offensive their raters found these items, against the size of the model's shift under that region:

| Region | Raters' offensive rate | Llama shift | Qwen shift |
|---|---|---|---|
| Arab Culture | 0.335 | +0.144 | +0.384 |
| Latin America | 0.305 | +0.216 | +0.381 |
| Sub-Saharan Africa | 0.286 | +0.246 | +0.462 |
| Indian Cultural Sphere | 0.266 | +0.085 | +0.382 |
| Western Europe | 0.227 | +0.135 | +0.386 |
| Sinosphere | 0.218 | +0.077 | +0.373 |
| North America | 0.205 | +0.102 | +0.351 |
| Oceania | 0.200 | +0.102 | +0.359 |

Spearman: **+0.62 (Llama), +0.67 (Qwen)**. Regions whose raters are stricter tend to get larger shifts. With 8 points this isn't significant, and it could reflect broad stereotypes about regions rather than genuine perspective. But it is the one place where framing lines up with real data at all.

### 10.3 Absolute distance: framing makes things worse
Mean absolute distance between model P(yes) and the framed group's own offensive rate, on items where that group has at least 4 raters:

| | Llama | Qwen |
|---|---|---|
| Unframed baseline | 0.447 | 0.313 |
| With the group's framing | **0.541** | **0.659** |

- **Every one of the 10 conditions in both models moves predictions further from that group's own ratings.**
- In Llama only 15-47% of items get closer. In Qwen it is 0-16%.
- The reason is section 1: models already over-flag, all real groups rate these items far lower (12-34% offensive), and framing pushes P(yes) up further.

**This is probably the most important finding for the thesis direction.** Demographic prompting doesn't approximate the group. It pushes the model further from every group, toward more flagging. It also gives a clear, measurable target for steering: reduce the distance to the framed group's ratings, which plain prompting increases.

---

## 11. Other covariates

- **Rater disagreement (entropy) barely predicts sensitivity.** Correlation with shift is −0.10 (Llama) and +0.15 (Qwen). Once baseline headroom is accounted for, high-disagreement items are not especially sensitive to framing. This supports the decision to stop selecting items by entropy.
- **Category:** in Llama, social-group items shift least (+0.085 against +0.12-0.14) because they start highest (ceiling). In Qwen there's no clear pattern.
- **Human fraction offensive** doesn't predict the shift (−0.13 and 0.00). The model's movement isn't tied to how offensive people found the item.

---

## 12. Llama versus Qwen

| | Llama | Qwen |
|---|---|---|
| Baseline over-flagging | strong (79%) | stronger (85%) |
| Items in the 0.02-0.98 band | 236 | 64 |
| Attention sink | 80% | 66% |
| Real framing shift, prefix | +0.06 to +0.25 | +0.35 to +0.46 |
| Placebo pool | about zero | clearly positive (+0.15) |
| Real minus placebo | +0.13 | +0.24 |
| Spread between demographics | wide | narrow |
| Muslim | no effect | normal |
| Suffix | everything down, content irrelevant | real framings down *more* than placebos |
| Confidence | placebos reduce it | everything increases it |

The direction of the core result is the same in both models. Real framings push toward offensive beyond placebos, prefix and reworded agree, suffix is confounded, and framing moves away from real groups. The size and texture differ a lot. Qwen is more extreme and more uniform, which matches its more extreme baselines. With only 64 items, Qwen's estimates are less precise.

---

## 13. Corrections to earlier claims

| Earlier claim | Status now |
|---|---|
| Placebo pushes the model the opposite way | **Only for the tea sentence.** Placebos are phrase-dependent. In Qwen most push upward. |
| Arab Culture is special | **Only in attention.** Behaviourally Sub-Saharan Africa and Latin America are larger. |
| Man, woman, Western Europe lose to the placebo on attention | **A length artifact.** Against same-length placebos, real phrases get more attention. |
| Uncertain items shift twice as much | **A headroom effect.** The shift scales with how low the baseline is. The real-placebo gap holds everywhere. |
| Attention share tells us how much a phrase matters | **Mostly not.** It tracks phrase length. The item-level link to behaviour is weak. |
| Layers 9-18, peak 13 | About the Muslim-Christian asymmetry in the old 30-item probe. **No layer data has been analysed for the current run.** |

---

## 14. Implications for steering

1. **There is a real, robust effect to steer.** Identity framing changes the decision beyond any neutral persona, in both models, in both clean structures, at matched lengths.
2. **There is a clear, measurable failure to fix.** Framing moves the model further from the framed group (MAE 0.45 → 0.54 in Llama, 0.31 → 0.66 in Qwen). Steering success criterion: **lower distance to the framed group's ratings than plain prompting**, ideally with sensitivity to *which items* that group rates differently (item-level correlation above zero).
3. **Head-averaged attention is not the right handle.** It mostly tracks phrase length. Before PASTA-style reweighting, the per-head data needs to show which heads, if any, respond to identity content beyond length. The obvious test: heads where real phrases get more attention than their exact token-matched placebos, and where that attention predicts the shift.
4. **More attention to the identity might make things worse.** PASTA increases attention to a span. Framing currently pushes predictions the wrong way. If more attention amplifies the same effect, steering would increase the distance to real groups. This needs to be checked directly, and it is a good hypothesis to test explicitly: "does upweighting the identity span bring the model closer to the group, or amplify the existing bias?"
5. **Use the controls built here in the steering evaluation:** steer on placebo spans too, compare against token-matched placebos, control for baseline P(yes), and use prefix or reworded only.
6. **Llama is the better first target.** It has wider item coverage (236 items), a near-zero placebo baseline, and a clearer spread between demographics. Qwen works as a replication.
7. **Muslim in Llama is a natural case study.** It is the one identity with no prompting effect, so it shows whether steering can create an effect where prompting doesn't.

---

## 15. Limitations

- One dataset, one prompt template family, two 7-8B models.
- Qwen's estimates rest on 64 items.
- Per-item p-values overstate certainty for phrase-level effects.
- The context block is confounded by the word "offensive".
- Group-level alignment uses 8 points.
- Attention is correlational. Nothing here shows the model *uses* the phrase through attention.
- Placebos are 6 phrases. They vary a lot, which is itself a finding, but 6 is still a small sample of possible neutral personas.

---

## 16. What to check when the current run lands

- **infix versus prefix versus suffix, same phrase and token length:** is there a gradient (prefix > infix > suffix) or a cliff at the suffix position? This separates "closer to the answer" from "after the instruction".
- **target ("offensive for a woman"):** does asking about impact on a group behave like self-identification? If target moves predictions closer to the group's ratings, that's a meaningful contrast with section 10.
- **Charts for items 899, 3969, 3263 (Llama):** check whether the demographic span's attention grows from prefix to suffix, and whether item-text attention changes.
- **Re-check that the prefix numbers here replicate in the new run** (same items, same code path), as a stability check.
- Then the per-head arrays.
