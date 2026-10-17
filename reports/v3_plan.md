# v3 plan: from "where does the identity effect live" to steering

v2 established the behaviour. Region framing has a real content effect in both models and every prompt structure. Gender, religion and the bare identity words do not stand out from content-free phrases of the same length. No framing brings the models closer to the real rater groups; it moves them further away.

v3 asks three questions in order:

1. **Where** in the model does the region information get picked up? (per-head attention, correlational, CPU)
2. **Does it matter causally?** If we switch those heads off or turn them up, does the effect shrink or grow? (GPU run B)
3. **Can steering do what prompting cannot?** Make the model follow the real groups, create an effect where prompting has none, or reduce over-flagging. (GPU run C)

Structures used throughout: the **prefix family** (prefix, reworded, imagine, consider), **infix** and **target**.
- The prefix family is the main set: four wordings of the same perspective framing, which agree almost perfectly behaviourally.
- Infix is the replication: same sentence, different position.
- Target ("offensive for X?") is the contrast case. It asks whether the text is about the group rather than taking the group's perspective. If the same heads carry both, that is a finding. If different heads carry them, so is that.

Same rules as v2: identical items, n and phrases for both models. Selection only on split A (250 items), causal tests only on split B (250 items). Every selection rule is written down before the data it is tested on is seen.

---

## Step 1. Per-head attention analysis (CPU, laptop, split A only)

Input: the two `master_grid_perhead_*.npz` files from v2. These hold raw and value-weighted attention from the answer position, per layer and head, to four segments (identity span, instruction, item text, other). No new GPU time.

1. **Clean the head set.** Flag "sink heads" that put almost all attention on the first token (for example more than 95% on average). Their attention to the identity span is noise, so they are excluded from the candidate search but still counted in the totals.
2. **Length diagnostic.** How strongly does each head's attention to the span follow its token length across the 22 conditions? v1 found head-averaged attention was mostly length. This puts a number on it per head.
3. **Content test per head.** Per variant: fit attention-per-token against length through the 8 placebos, then ask whether a region phrase gets more (or less) attention than a new content-free phrase of its length would. This is the same length-adjusted null that worked in v2, now per head, with FDR correction across heads. Run once on raw and once on value-weighted attention.
4. **Behaviour link per head.** Across items in split A: when a head gives the region phrase more attention than its matched placebos, is the region-specific shift in log-odds larger too? Spearman per head, FDR-corrected.
5. **Stability.** Split A into two random halves. Do the same heads rank high in both? Do they replicate across the prefix family, infix and target? Do the two models use heads at a similar relative depth?
6. **Freeze the candidate list.** Rule: passes the content test for regions in at least 2 of the 3 structure groups, passes the behaviour link, is stable across halves, and is not a sink head. The rule and the resulting list go into a file before run B starts.

Figures: head heatmaps (content t, behaviour rho) per model and structure; layer profiles; head-averaged vs per-head, to show what averaging hides.

**What I expect.**
- Most heads fail the content test once length is controlled. Head-averaged attention was mostly length in v1, and v2 showed content-free phrases differ a lot among themselves.
- A small set of mid-layer heads (v1 pointed at layers 9 to 18) should pass for regions in the prefix family and infix.
- Behaviour links will be weak (rho around 0.1 to 0.2 at best), because the answer depends on much more than one head.
- Value-weighted attention will reduce the sink's share a lot. This is the known Kobayashi result, and it already showed in the local test. The candidate lists from raw and value-weighted attention will overlap, but not fully.
- Target will most likely recruit partly different and later heads, since it is a "does the text mention X" question.

## Step 2. Run B: causal tests (GPU, COMA, split B only)

Scripts already exist and are tested on a small model: `ablation_heads.py` and `head_profiling.py`. Before running them I'll update them for regions and the three structures.

1. **Ablation and amplification of the candidates.** Multiply the candidates' attention to the identity span by 0 (removed), 2 or 4, at the answer position. Measure the region-specific shift (region minus its exact-length placebos) on split B.
2. **Controls.**
   - 10 random head sets with the same number of heads per layer. This gives a distribution, so "the candidates matter more than random heads" becomes a percentile, not a guess.
   - The matched placebos under the same intervention.
3. **Upper bounds.**
   - All heads at the answer position only. Tells us whether the region information is read directly at the end, or already absorbed into earlier tokens.
   - All heads at all positions. A sanity check: if no token can see the phrase, the effect must disappear.
4. **PASTA head profiling (split A).** Steer one head at a time toward the span (PASTA's alpha = 0.01 on everything else) and rank heads by how much the region-specific answer changes. This is causal head selection, the way the PASTA paper does it, and a cross-check on step 1.
5. **Side effects.** KL divergence of the full next-token distribution, and agreement with the human majority label before and after.

Cost estimate (H200, free on COMA): ablation about 1 to 1.5 h per model, profiling about 1.5 to 3 h per model depending on scope. One or two jobs per model.

**What I expect.**
- Removing all attention to the phrase at all positions removes the effect. That's guaranteed by construction and only verifies the setup.
- Removing it at the answer position only will probably leave a large part of the effect. In v2's charts, attention went to "You" and the final period rather than to the identity words, which suggests the information is gathered into intermediate tokens first. If so, the steering has to act on more than the last position.
- The correlational candidates may or may not beat random heads. Both outcomes are reportable: "attention patterns explain the behaviour" or "they don't".
- Profiling will find a short list of heads with a clearly larger causal effect than the rest. Its overlap with the correlational list is an open question.

## Step 3. Go/no-go and a written steering protocol (CPU)

Before any steering run I write down:
- target heads (from profiling, the correlational list, or both), demographics, alpha values;
- the success metrics with their direction;
- the baselines.

Then a supervisor check-in with the evidence from steps 1 and 2.

Go if: some heads carry the region effect causally (candidates beat at least the 95th percentile of the random sets, or profiling finds clearly separated heads), and the answer-position upper bound shows that attention carries a meaningful part of the effect.

No-go is still a result: "the identity effect does not run mainly through attention to the phrase". In that case steering moves to activation steering (see step 4, arm D), which does not depend on attention.

## Step 4. Run C: steering (GPU, COMA, split B)

New script: `steer_eval.py`, which applies PASTA steering during the forward pass and scores the outcome. Four arms:

| arm | what is steered | question | metric that would count as success |
|---|---|---|---|
| A. amplify region | profiled heads, toward the region span | does more attention to the region mean more region effect? (dose-response) | region-specific shift rises with alpha, more than for random heads and placebo spans |
| B. create an effect | same heads, toward gender / religion spans | can steering produce an effect prompting alone does not? | gender/religion shift rises beyond same-length placebos under the same steering |
| C. toward the real group | region and gender spans, with the change in distance to the real rater group as the target | can steering move the model toward the group's actual ratings? | lower distance to the group's offensive rate than unsteered framing, and item-level tracking of group gaps on held-out items |
| D. toward the text | the item text span instead of the identity | does more attention to the text reduce over-flagging? | closer to the human majority label, fewer false "offensive" |

Baselines for every arm:
- prompt only (`context_long`, already measured);
- random heads with the same layer mix;
- steering toward a placebo span;
- activation steering (difference-of-means direction between framed and unframed hidden states), to separate "attention steering does something" from "any intervention does something".

**What I expect, honestly.**
- **Arm A** will work: more attention to the region phrase should mean a larger region effect. That also means further from the real groups, since the existing effect points away from them. This is the clean causal demonstration.
- **Arm B** is uncertain. Steering might inflate gender and religion the same way it inflates placebos, which would then not count.
- **Arm C** is the hardest. No current condition moves the models toward the groups, and attention steering has no direct way to know which way a group leans. A positive result here would be the strongest finding of the thesis. A negative one ties back cleanly to v2.
- **Arm D** is the one that speaks to your concern about LLMs for offensive-language detection. The models over-flag, and the item text gets only about 13% of content attention. If steering toward the text reduces false positives, that is a constructive result independent of demographics. It's speculative, which is why it's a separate arm.

## Step 5. Write-up

- Update `metrics_glossary.md` with the new measures: length-adjusted null, phrase-level p, value-weighted attention, ablation measures, KL.
- Rewrite the proposal's evidence table with the v2 numbers.
- Methods text: the exact prompt per variant, item selection, matching, and statistical units.
- Limitations:
  - the items are borderline for both models;
  - 2 phrases per axis for gender, religion and identity;
  - attention is not explanation;
  - the answer is a single 0/1 token;
  - English only;
  - no rater religion in D3CODE.

---

## Decisions taken (8 Oct 2026)

- Regions: 4 instead of 8, the same for both models. **Sub-Saharan Africa** (strongest in both models), **Arab Culture** (the v1 continuity region), **Western Europe** (the Western reference) and **Latin America** (where the models disagree most: strong in Llama, not significant in Qwen). That is two 4-token and two 6-token nouns.
- Arm D (steering toward the text): in.
- Activation-steering baseline: in.

## Direction confirmed by the supervisor (10 Oct 2026)

**Success criterion for steering: getting closer to the real people's average rating.** Any outcome is reportable at this stage. In numbers:

- "Real average" = the share of raters who called the text offensive (`rating_binary`, D3CODE):
  - for a framed prompt, the raters from that group (region or gender), on texts with at least 4 such raters;
  - for an unframed prompt, and for steering toward the text (arm D), all raters.
- Distance = mean absolute difference between the model's P(offensive) and that share, over the held-out texts (split B).
- Steering "succeeds" when the steered distance is lower than the unsteered distance of the same prompt. It also has to beat the two controls: steering random heads with the same layer mix, and steering toward a placebo span.
- Reference points from v2 (prefix, all items): unframed 0.35 (Llama) / 0.40 (Qwen); framed 0.51 / 0.68.

## Blockers and decisions

**Technical**

1. **The per-head files exist only on your laptop.** Step 1 runs locally, not in a cloud session. Run B and C run on COMA, which has its own copy.
2. **Code to update before step 1:** `analysis_perhead_candidates.py` still has the v1 setup (placebo null only under prefix, one matched placebo, 4 fixed demographics). It needs: the length-adjusted null per variant, the six structures, split A only, raw and value-weighted, sink-head exclusion, split-half stability, and the freeze rule. About half a day of my work, CPU only.
3. **Code to update before run B:** region conditions and the three structures as defaults in `ablation_heads.py` and `head_profiling.py`; an all-positions option for the upper bound; the side-effect measures against the human labels.
4. **New code before run C:** `steer_eval.py` with the four arms and the activation-steering baseline. This is the largest new piece. I'd write it only after step 3, because its targets depend on runs A and B.

**Decisions for you**

1. **Which regions to steer.** All 8, or a subset such as the strongest (Sub-Saharan Africa), the most "Western" (Western Europe) and one in between? All 8 costs more GPU time, but gives a dose-response across regions.
2. **Arm D (steering toward the text).** In or out? It's the arm that addresses detection quality directly, but it's further from the thesis title.
3. **Activation-steering baseline.** It strengthens the conclusions a lot, but it's real extra work. In or out?
4. **Supervisor check-in.** After step 2, or only after step 3? I'd do it after step 2, with the causal pilot in hand.

**Things that could change the plan**

- If no head passes step 1 for regions, step 2 runs on the profiling heads alone. That was always the stronger method.
- If the answer-position upper bound shows that attention at the end carries almost nothing, steering has to act at all positions. That is closer to PASTA's original setting anyway.
- If the causal tests fail completely, steering switches to activation steering, and the attention result becomes "attention patterns don't carry the identity effect".
