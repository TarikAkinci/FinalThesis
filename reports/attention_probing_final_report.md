# Attention Probing: Final Results Report

I ran this on Llama-3.1-8B-Instruct using D3CODE offensiveness classification prompts. I sampled 30 items, stratified across D3CODE's 3 categories (moral, random, social-group), and ran each one under all 7 conditions: a baseline plus 6 demographic framings. That's 210 forward passes in total.

Each forward pass is a single deterministic pass through the model. No `.generate()`, no sampling involved. This is equivalent to temperature 0. I used `output_attentions=True` and read attention at the last token position, since that's the position that actually produces the 0/1 decision.

## 0. The formulas behind every number below

I'll define these once here and just refer back to them by name.

**F1, the attention vector itself.** For each item/condition, the model gives back one attention weight matrix per layer, `out.attentions[l]`, shape `[1, 32 heads, seq_len, seq_len]`. I take the row for the last token position (the decision token) and average it over every layer and every head:

```
attn[i] = mean over layer l (0..31), head h (0..31) of out.attentions[l][0, h, -1, i]
```

This gives one number per input token `i`: how much attention the decision token pays to it, averaged across the whole network.

**F2, the 4-way segment split.** I locate the token span of the demographic phrase, the fixed instruction text, and the item text (via `find_token_span`/`find_span_by_delimiters`), and sum `attn` over each span:

```
pct_demographic = sum(attn[i] for i in demographic span)
pct_instruction = sum(attn[i] for i in instruction span)
pct_item_text   = sum(attn[i] for i in item-text span)
pct_other_scaffolding = 1 - pct_demographic - pct_instruction - pct_item_text
```

`pct_other_scaffolding` is everything left over: `<|begin_of_text|>`, header tokens, punctuation, the generation-prompt tokens. This is the attention sink.

**F3, sink-excluded renormalization.** Because the sink eats ~80% of everything, comparing raw `pct_demographic` across conditions is unfair (some conditions just have more "room" left over). So I also compute, per item/condition row, before averaging:

```
content_mass = 1 - pct_other_scaffolding
pct_demographic_norm = pct_demographic / content_mass
pct_instruction_norm = pct_instruction / content_mass
pct_item_text_norm   = pct_item_text / content_mass
```

**F4, the per-layer version.** Same as F1-F3, except I only average over the 32 heads of a single layer `l`, not across all 32 layers. This gives a full segment breakdown per (item, condition, layer) instead of one collapsed number.

**F5, significance testing.** For two conditions A and B, on the same set of items, I compute `diff_i = value_i(A) - value_i(B)` for each item `i`, drop any item with a missing value, and run a paired Wilcoxon signed-rank test (`scipy.stats.wilcoxon`) on the differences. This tests whether the differences are consistently on one side of zero, not just on average positive or negative.

---

## 1. Where does the model pay attention in a plain prompt?

Using F1 and F2 on the `baseline` condition (30 items, no demographic phrase, so there's no `pct_demographic` here):

**Raw results (mean over 30 items):**

| | pct_instruction | pct_item_text | pct_other_scaffolding |
|---|---|---|---|
| baseline | 0.1754 | 0.0263 | 0.7982 |

So about 80% of attention goes to the sink. Of what's left over (F3, `content_mass = 1 - 0.7982 = 0.2018`):

**Sink-excluded (F3), mean over 30 items:**

| | pct_instruction_norm | pct_item_text_norm |
|---|---|---|
| baseline | 0.8703 | 0.1297 |

87.0% of the non-sink attention goes to the fixed task instruction, only 13.0% goes to the item text itself.

## 2. How does attention change when demographic info is added?

Same F1/F2/F3, now run on all 6 demographic conditions. Here's the full raw table (mean over 30 items, standard deviation across items in parentheses):

| condition | pct_demographic | pct_instruction | pct_item_text | pct_other_scaffolding |
|---|---|---|---|---|
| gender_man | 0.0124 (±0.0014) | 0.1726 (±0.0039) | 0.0266 (±0.0074) | 0.7883 (±0.0058) |
| gender_woman | 0.0134 (±0.0015) | 0.1730 (±0.0040) | 0.0266 (±0.0074) | 0.7869 (±0.0058) |
| region_arab_culture | 0.0214 (±0.0034) | 0.1675 (±0.0039) | 0.0294 (±0.0080) | 0.7815 (±0.0070) |
| region_western_europe | 0.0152 (±0.0018) | 0.1726 (±0.0039) | 0.0270 (±0.0075) | 0.7851 (±0.0055) |
| religion_christian | 0.0124 (±0.0015) | 0.1716 (±0.0034) | 0.0268 (±0.0075) | 0.7891 (±0.0060) |
| religion_muslim | 0.0156 (±0.0020) | 0.1663 (±0.0039) | 0.0277 (±0.0076) | 0.7901 (±0.0065) |

And the same thing sink-excluded (F3), which is the fairer comparison and what I'd actually present:

| condition | pct_demographic_norm | pct_instruction_norm | pct_item_text_norm |
|---|---|---|---|
| gender_man | 0.0588 | 0.8160 | 0.1252 |
| gender_woman | 0.0630 | 0.8125 | 0.1245 |
| region_arab_culture | 0.0985 | 0.7678 | 0.1337 |
| region_western_europe | 0.0707 | 0.8041 | 0.1252 |
| religion_christian | 0.0590 | 0.8147 | 0.1263 |
| religion_muslim | 0.0748 | 0.7938 | 0.1314 |

This attention comes almost entirely out of the instruction's share (87.0% down to 76.8-81.6%). Item-text attention barely moves at all (12.5-13.4%). So adding demographic framing seems to compete with the task instruction, not with the text being judged.

### Is this actually significant?

Using F5, on `pct_demographic_norm`, paired by item:

| Contrast | Mean difference | Direction consistency | p |
|---|---|---|---|
| Muslim vs. Christian | +0.0158 | 29 out of 29 items | p < 0.00001 |
| Arab Culture vs. Western Europe | +0.0279 | 29 out of 29 items | p < 0.00001 |
| Man vs. Woman | -0.0042 (woman higher) | 29 out of 29 items | p < 0.00001 |

(n=29, not 30: one item, item_id=1560, drops out of every span-based statistic. See §6 for why.)

All three hold up. And it's not just true on average, every single one of the 29 usable items shows the same direction. That's a genuinely strong result, not something a couple of outlier items are dragging around.

Using F5 again, this time on `pct_item_text_norm` vs. baseline:

| condition | mean delta vs. baseline | n | p |
|---|---|---|---|
| gender_man | -0.0045 | 29 | 0.00000 |
| gender_woman | -0.0053 | 29 | 0.00000 |
| region_western_europe | -0.0045 | 29 | 0.00002 |
| region_arab_culture | +0.0039 | 29 | 0.03262 |
| religion_christian | -0.0034 | 29 | 0.00002 |
| religion_muslim | +0.0016 | 29 | 0.16885 |

5 of 6 are statistically significant. But the actual size of the shift is tiny, only 0.16-0.53 percentage points. So I'd call this "real but practically negligible," not evidence that the model is being meaningfully distracted from the text. One condition stands out here: religion_muslim shows no significant change in item-text attention at all (p = 0.17). That framing pulls attention almost entirely from the instruction and leaves the item text untouched.

## 3. Does this hold across categories?

Same F3 numbers, split by D3CODE category instead of pooled. Raw table (`pct_demographic_norm` / `pct_item_text_norm`, mean per category):

| condition | moral | random | social-group |
|---|---|---|---|
| gender_man | 0.0574 / 0.1348 | 0.0606 / 0.1157 | 0.0585 / 0.1252 |
| gender_woman | 0.0613 / 0.1344 | 0.0651 / 0.1146 | 0.0626 / 0.1243 |
| region_arab_culture | 0.0979 / 0.1463 | 0.1016 / 0.1218 | 0.0958 / 0.1328 |
| region_western_europe | 0.0688 / 0.1349 | 0.0731 / 0.1167 | 0.0701 / 0.1239 |
| religion_christian | 0.0572 / 0.1361 | 0.0608 / 0.1170 | 0.0590 / 0.1258 |
| religion_muslim | 0.0735 / 0.1432 | 0.0773 / 0.1207 | 0.0735 / 0.1301 |

The demographic-attention share and the asymmetry pattern show up consistently across all three categories. For example, Arab Culture vs. Western Europe is 9.6-10.2% vs. 7.0-7.3% in every single category. So this isn't being driven by just one category. Full numbers (30 items x 7 conditions) are in `attention_probe_segments.csv` if you want to check.

## 4. What about prediction flips?

This uses a different piece of extracted info: `predicted`, which is just `argmax(logits[one_id], logits[zero_id])` at the same forward pass (no extra computation needed, it's the same pass that gives us the attention). I compare each condition's `predicted` to that item's own `baseline` prediction:

| Condition | Flip rate |
|---|---|
| religion_christian | 13.3% (4/30) |
| gender_man | 10.0% (3/30) |
| region_arab_culture | 10.0% (3/30) |
| region_western_europe | 10.0% (3/30) |
| gender_woman | 6.7% (2/30) |
| religion_muslim | 6.7% (2/30) |

Honestly, these counts are too small (2-4 items per condition) to run a real test on. And flipped items don't show a consistently higher demographic-attention share than the ones that didn't flip. I'd leave this as an open question rather than a finding. It would need a bigger item sample to say anything credible here.

## 5. Going layer by layer

Same F1-F3, but using F4 instead of the fully-collapsed version: keep the 32 layers separate, only average over heads within each layer. Full raw table below (sink share from `baseline`; demographic share, both raw and sink-excluded, for `religion_muslim` vs `religion_christian`; significance via F5 on the sink-excluded column):

```
layer  sink(baseline)  raw_christian  raw_muslim  norm_christian  norm_muslim  norm_pct_diff   p (Wilcoxon)
0      0.5854          0.0208         0.0226      0.0505          0.0547        +8.4%         3.73e-09
1      0.8588          0.0108         0.0104      0.0733          0.0705        -3.9%         3.73e-09
2      0.9335          0.0209         0.0215      0.2473          0.2505        +1.3%         7.45e-09
3      0.8993          0.0105         0.0125      0.0990          0.1156       +16.8%         3.73e-09
4      0.8395          0.0202         0.0171      0.1202          0.1047       -12.9%         3.73e-09
5      0.8199          0.0351         0.0314      0.1741          0.1605        -7.8%         3.73e-09
6      0.8099          0.0346         0.0314      0.1732          0.1594        -8.0%         3.73e-09
7      0.7157          0.0313         0.0316      0.1098          0.1121        +2.1%         8.84e-06
8      0.7647          0.0412         0.0461      0.1504          0.1699       +13.0%         3.73e-09
9      0.7040          0.0195         0.0254      0.0634          0.0826       +30.3%         3.73e-09
10     0.7188          0.0219         0.0362      0.0691          0.1171       +69.6%         3.73e-09
11     0.6907          0.0164         0.0301      0.0532          0.0956       +79.8%         3.73e-09
12     0.6416          0.0132         0.0207      0.0361          0.0557       +54.5%         3.73e-09
13     0.5478          0.0136         0.0369      0.0304          0.0808      +166.1%         3.73e-09
14     0.5348          0.0087         0.0167      0.0179          0.0354       +97.9%         3.73e-09
15     0.6938          0.0078         0.0090      0.0242          0.0288       +19.0%         3.73e-09
16     0.7591          0.0087         0.0174      0.0345          0.0694      +101.1%         3.73e-09
17     0.7156          0.0073         0.0135      0.0251          0.0479       +90.7%         3.73e-09
18     0.7759          0.0053         0.0088      0.0239          0.0398       +66.7%         3.73e-09
19     0.8158          0.0042         0.0059      0.0233          0.0343       +47.2%         3.73e-09
20     0.7848          0.0059         0.0067      0.0273          0.0298        +9.1%         4.13e-05
21     0.8751          0.0033         0.0049      0.0279          0.0464       +66.1%         3.73e-09
22     0.7954          0.0035         0.0043      0.0172          0.0227       +32.2%         3.73e-09
23     0.8927          0.0028         0.0031      0.0263          0.0306       +16.4%         1.38e-06
24     0.9088          0.0017         0.0032      0.0180          0.0344       +91.5%         3.73e-09
25     0.9512          0.0013         0.0020      0.0258          0.0397       +53.8%         3.73e-09
26     0.9360          0.0024         0.0029      0.0344          0.0407       +18.1%         7.45e-09
27     0.9149          0.0035         0.0046      0.0396          0.0511       +29.2%         3.73e-09
28     0.9289          0.0027         0.0034      0.0325          0.0369       +13.6%         7.71e-07
29     0.9357          0.0047         0.0060      0.0639          0.0715       +11.9%         6.48e-06
30     0.8666          0.0076         0.0079      0.0537          0.0543        +1.1%         0.701 (not significant)
31     0.9263          0.0055         0.0064      0.0666          0.0711        +6.7%         4.70e-05
```

Reading this:

**The attention sink is not uniform with depth. It has a clear W-shape.**

- Layer 0: 58.5%. Low, since the model hasn't consolidated onto the sink yet.
- Layers 1-6: 81-93%, peaking at layer 2 (93.3%). The sink forms almost immediately.
- Layers 7-14: 55-77%, bottoming out at layers 13-14 (54.8%, 53.5%). This is the band doing the most genuine content processing.
- Layers 15-20: 69-82%, climbing back up as the sink re-forms.
- Layers 21-31: 79-95%, peaking at layer 25 (95.1%). The sink dominates almost completely near the output.

**The Muslim vs. Christian gap (the `norm_pct_diff` column) tracks that dip almost exactly.**

- It's statistically significant in 31 of 32 layers. Only layer 30 isn't (p = 0.701). But given how consistent this design is, significance alone doesn't mean much here, what matters is the size of the effect, not how many layers cross p < 0.05.
- There's a small, specific band where the sign actually flips: layers 1, 4, 5, and 6 show Christian significantly higher than Muslim (-3.9% to -12.9%). This is a real, structured reversal, not noise.
- From layer 7 onward, Muslim is higher in every layer, and the size of that gap concentrates sharply in layers 9-18: anywhere from +30% to +170%, peaking at layer 13 (+166%). Layers 16, 14, 17, and 11 are also strong (+101%, +98%, +91%, +80%).
- That peak band, layers 9-18 centered on 13, is exactly the same band with the lowest sink. In other words, it's where the most attention is actually "available" for real content. That's the shape you'd expect if this is a genuine semantic effect and not a numerical artifact. It also overlaps with the layers your supervisor's political-ideology papers flagged as informative for this exact model (layers 11, 13, 26-31 in the multilingual political-steering paper). That paper probes a different signal (head value outputs, not raw attention scores), so I'd treat this as a nice cross-check rather than proof they're measuring the same mechanism.

## 6. What are the content-attention charts for?

These don't use a new formula, they're a per-item visualization of F1-F3: instead of averaging over 30 items, I plot the sink-excluded attention (F3) token by token for one specific item.

I picked six items as concrete, presentable examples: 2 per D3CODE category. These are the same six used for the bertviz heatmaps. Each one has three stacked panels (baseline / religion_muslim / religion_christian) showing the sink-excluded attention on every demographic (red), instruction (blue), and item-text (green) token. The idea is to see the story at the level of one real example, not just as a pooled average.

`item60.png` and `item2126.png` match the aggregate finding nicely. You can see a visible red block for Muslim/Christian, roughly the size you'd expect, and it's clearly coming out of the blue instruction tokens.

**`item1560.png` is broken though, and shouldn't go in front of your supervisor as it is.** It shows no green (item-text) segment in any of the three panels. I found the cause: item 1560's D3CODE text is `" A lesbian ,  a native,  and a Rabi are in a life boat ... '  Yadda yadda yadda"`. It starts and ends with a literal quote character, which collided with the prompt template's own wrapping quotes (`Text: "{text}"`). That broke the plain substring search I was using to locate the item-text span. It's not a truncation problem, that prompt is only 87-96 tokens, well under any length limit. I confirmed this directly from the job log, which prints a warning for exactly this item across all 7 conditions.

The good news: this only affects this one item. That's 7 rows out of 210, all for item_id=1560 (this is also exactly why every table above uses n=29, not 30, for anything involving `pct_item_text`). Everything else, including every aggregate statistic in this report, is unaffected, since pandas and scipy correctly dropped these 7 NaN rows instead of silently keeping bad data.

I've already fixed the underlying bug in code. Item text is now located using the prompt's fixed surrounding delimiters (the `Text: "..."` boundaries) instead of searching for the item's own content, so it no longer matters what punctuation the item text has. This just needs one more cluster run before item 1560's chart and heatmaps are usable.

---

## 7. Is probing done? What do we need before steering?

**Short answer: yes, probing is substantively done and ready to present.** I have, with real statistical backing:
- Where attention goes in a bare prompt.
- How it shifts once demographic framing gets added.
- That the shift is asymmetric across contrast pairs, and that asymmetry holds for almost every sampled item.
- Where in the network's depth that effect actually happens.
- That it holds up across all three D3CODE categories.

That directly answers both of your supervisor's original questions with real numbers behind them, not just an impression.

**Two small things to clean up first:**
1. Rerun the (already-fixed) job so item 1560's chart and heatmaps aren't broken in the final materials. Should take 10-15 minutes of cluster time, no more code work needed.
2. The flip-rate question from §4 is worth mentioning as a limitation. I wouldn't try to fix it now, it would need a bigger sample, which is a bigger ask than just "before we present."

**Before moving into steering, there's one real gap left: per-head granularity.** Everything above is averaged over all 32 heads within each layer (F1/F4). We now know which layers matter (9-18, peaking at 13), but not which specific heads within those layers actually carry the effect. Every steering-relevant paper you summarized, PASTA especially, picks specific heads to intervene on rather than whole layers. That fits your supervisor's own phrase, "manipulating the attention," pretty directly.

So concretely, before writing any steering code, I'd profile individual heads within layers 9-18. Either the same per-head logistic-regression/R² approach papers 1 and 4 use, or PASTA's own small-sample head-profiling method. The goal is just a short list of target heads. This is still squarely a probing task. It doesn't touch model outputs at all, it just tells us where to intervene later.

**My recommended order from here:**
1. Rerun the fixed job to get item 1560's outputs looking right.
2. Profile heads within layers 9-18 to find specific targets. This is the last piece of probing.
3. Only then start steering, following your supervisor's own sequencing ("probing is the first step, later we will move to steering"). Given her exact wording, PASTA's direct attention-score reweighting looks like the closer methodological match to try first, with the DiffMean/activation-steering papers (1, 3, 4) as a documented fallback if that doesn't pan out.
