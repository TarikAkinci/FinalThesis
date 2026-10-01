# Metrics glossary

Every metric used across the zero-shot, attention-probing and master-grid scripts, with its exact formula (checked against the code, not from memory), how it's measured, and how to read it. Grouped by what it describes.

---

## 1. Human rating metrics (from D3CODE)

These come straight from `d3-ratings.csv`, before any model is involved.

### `rating_raw`
The individual rater's score on D3CODE's 0-4 offensiveness scale (0 = not offensive, 4 = most offensive). One row per rater per item. `-1` means "didn't understand" and is dropped, not treated as a rating.

### `rating_binary`
D3CODE's own binary collapse of `rating_raw` (offensive / not offensive). Used wherever we need a single "did this rater call it offensive" flag, e.g. group rating gaps.

### `entropy` (human, item-level)
Shannon entropy of the empirical distribution of `rating_raw` across the 0-4 scale for one item, in bits:
```
entropy = -sum(p_v * log2(p_v))  for v in {0,1,2,3,4}, p_v = fraction of raters who gave rating v
```
Computed by `evaluate.py`'s `ratings_to_distribution` + `entropy`. **Range 0 to log2(5) ≈ 2.32.** 0 means every rater picked the same value (no disagreement); the maximum means ratings are spread evenly across all 5 values (maximum disagreement). This is the "high-disagreement" ranking used throughout, and it's now recorded as a per-row covariate in the master grid rather than a selection filter.

### `variance`
Plain variance of `rating_raw` across raters for one item. A second, cruder disagreement measure alongside entropy; kept in the item files but entropy is what selection and reporting use.

### `p_offensive`
Mean of `rating_binary` across all raters for one item, i.e. the fraction of raters who called it offensive. This is the human ground truth the model's `prob_yes` is compared against.

### `n_raters`
How many raters rated that item. Used as a reliability floor (`MIN_RATERS`) before trusting an item's entropy.

### Group rating gap
Per item, per demographic group (e.g. Region = "Arab Culture"): `group_mean_rating_binary - everyone_else_mean_rating_binary`, computed only when the group has at least 4 raters on that item. Positive means that group rates this item more offensive than the rest of the raters do. This is what the model's per-condition shift is checked against in the "does the shift track real group opinion" test.

---

## 2. Model prediction metrics

These come from one deterministic forward pass per prompt (`output_attentions=True`, no sampling, equivalent to greedy decoding at temperature 0).

### `prob_yes` (also called P(yes) or `prob_one`)
```
prob_yes = softmax(logits[[token_id("0"), token_id("1")]])[1]
```
The model's probability that the very next token is "1" rather than "0", read directly off the logits, not from a generated string. **Range 0 to 1.** This is the model's continuous confidence that the text is offensive, and it's the primary outcome variable everywhere in the grid.

### `predicted`
`"1"` if `logits["1"] > logits["0"]`, else `"0"`. The thresholded (P(yes) ≥ 0.5) binary call, i.e. what the model would actually answer.

### `baseline_prob_yes` / `baseline_pred`
The same two values, computed once per item with no demographic phrase at all (plain prompt). Every conditioned row carries its own item's baseline alongside it, so shifts are always relative to that specific item's unframed behaviour.

### `delta_prob_yes` (the shift)
```
delta_prob_yes = prob_yes - baseline_prob_yes
```
**Signed**, range -1 to +1. Positive means the framing pushed the model toward calling the text offensive; negative means it pushed the other way. This is the number behind "mean signed shift" everywhere in the reports, and it's the one that matters, see the signed-vs-unsigned point below.

### `flipped_vs_baseline` / flip rate
Whether `predicted != baseline_pred`, i.e. whether the framing changed the actual yes/no answer, not just the probability. Flip rate = fraction of items where this happened. A much coarser signal than `delta_prob_yes` since it collapses e.g. a 0.02 shift and a 0.48 shift into the same "no flip" bucket unless it crosses 0.5.

### `mean_abs_delta` (unsigned / absolute shift)
```
mean_abs_delta = mean(|delta_prob_yes|)
```
Measures how much a condition moves predictions, ignoring direction. **This is the statistic that hid the placebo reversal**: a placebo pushing hard toward "not offensive" and a demographic framing pushing hard toward "offensive" can have nearly identical `mean_abs_delta`, making them look equally (in)significant, when they are actually opposite and both real. Always read `mean_abs_delta` next to `mean_signed_delta`, never alone.

### `items_up` / `items_down` / `items_flat`
Count of items where `delta_prob_yes` is positive / negative / exactly zero, for one condition. The plain-language version of the signed shift: "71 items pushed toward offensive, 23 pushed the other way" is more interpretable to a non-technical reader than a mean.

### `MIN_BASELINE_PROB` / `MAX_BASELINE_PROB` band (0.02 to 0.98)
Not a result metric but a selection rule: only items whose **baseline** P(yes) falls in this range are put through the condition grid. An item already at baseline P(yes)=0.999 has almost no room to move further toward "offensive", so any perturbation on it is mechanically biased downward regardless of content. This band exists specifically to remove that ceiling-effect confound.

---

## 3. Confidence / uncertainty metrics

### `margin`
```
margin = |prob_yes - 0.5|
```
**Range 0 to 0.5.** How far the model's probability sits from the undecided point. 0 = maximally uncertain (50/50), 0.5 = maximally confident (P(yes) is 0 or 1).

### `pred_entropy` (model prediction entropy, `binary_entropy`)
```
binary_entropy(p) = -(p*log2(p) + (1-p)*log2(1-p)),  p clipped to [1e-12, 1-1e-12]
```
Same formula family as the human rating entropy above, but applied to the model's own two-way probability (P(yes) vs P(no)) instead of a 5-way human rating distribution. **Range 0 to 1 bit.** 0 = model is completely certain either way; 1 = model is maximally torn (P(yes)=0.5). This is the model's own internal uncertainty, directly comparable in kind (though not scale) to the human entropy metric.

### `mean_delta_margin` / `mean_delta_entropy`
The condition's margin or entropy minus the same item's baseline margin/entropy, averaged. A **negative** `delta_entropy` means the framing made the model *more* confident than it was unframed; **positive** means *less* confident. This is the metric that separated real demographics (which made Llama more confident, negative delta) from the placebo (which made it less confident, positive delta) independent of which direction the prediction moved.

### Model confidence vs. human agreement (project-level check, not in `master_grid.py`)
`model_conf = |prob_yes - 0.5| * 2` (rescaled to 0-1) compared against `human_agree = |p_offensive - 0.5| * 2`, both per item. Used to test whether the model is *more* confident exactly where humans disagree most (it is: 0.93 vs 0.77 confidence on the most-contested vs most-agreed thirds of items).

---

## 4. Attention metrics

All computed at the **last token position** (the position the model uses to decide the next token) in a single forward pass, averaged over the model's layers and heads unless a per-layer or per-head breakdown is explicitly kept.

### Segment shares (`pct_demographic`, `pct_instruction`, `pct_item_text`, `pct_other_scaffolding`)
The attention weight (softmax output, sums to 1 across the whole sequence) summed over the token span belonging to each labeled region of the prompt:
- **demographic**: the inserted framing phrase (or its elaboration, for the context block)
- **instruction**: the fixed "Is this text offensive? Answer with only 0 (no) or 1 (yes)." text
- **item_text**: the actual text being classified
- **other_scaffolding** (the "sink"): everything else, i.e. `1 - (demographic + instruction + item_text)`. This is dominated by fixed chat-template tokens (`<|begin_of_text|>`, role markers, etc.) that receive disproportionate attention at essentially every layer regardless of content, a known phenomenon (Xiao et al. 2023's "attention sink"), not something specific to this task.

Each share is **raw attention mass**, range 0 to 1, and the four shares for one row sum to exactly 1.

### `_norm` suffix (sink-excluded / renormalized shares)
```
content = 1 - pct_other_scaffolding
pct_X_norm = pct_X / content
```
Re-expresses demographic/instruction/item-text shares as a fraction of only the **non-sink** attention, since raw shares are dominated by however much sink that particular row happened to have (78-85%, but it varies by layer and condition). `pct_demographic_norm` is the number actually compared across conditions and used in every attention chart and headline claim, raw shares are reported mainly as context.

### Per-layer attention
The same four shares, but kept separate per transformer layer (32 layers for Llama, 28 for Qwen) instead of averaged across all layers. Reveals that raw demographic attention peaks around layers 5-8 and decays afterward, and that sink dominance itself varies sharply by layer (as low as ~54% at layer 13-14, over 90% in some late layers), which is why layer-level claims must use the sink-excluded share, not the raw one.

### Per-head attention
The same shares kept separate per attention head within each layer (`master_grid.py`'s `.npz` output, shape rows x layers x heads x 4 segments). Not yet analyzed; this is the granularity needed before any PASTA-style steering, since steering reweights specific heads, not whole layers.

---

## 5. Length / matching metrics

### `phrase_words`
Word count of the inserted phrase (`len(phrase.split())`). The original, coarser length control, kept as a covariate.

### `phrase_tokens`
Token count of the phrase under the **specific model's own tokenizer** (`len(tok(phrase, add_special_tokens=False)["input_ids"])`). Llama and Qwen tokenize the same English phrase differently, so this is computed at runtime per model. This is the length measure actually used for matching, since two phrases with the same word count can still be different numbers of sub-word tokens.

### Exact token-matched placebo
For a handful of demographic phrases, a placebo phrase is chosen from a pool such that `phrase_tokens` is **identical** under that model's tokenizer (not just close). Comparing a real framing against its exact match removes length as a possible explanation for any remaining difference. When no exact match exists, the nearest available length is used and flagged (`NEAREST` in the run log) rather than silently substituted.

---

## 6. Statistical tests

### Wilcoxon signed-rank test (`p_delta_vs_zero`, `p_delta_vs_legacy_placebo`, `p_delta_vs_matched`, `p_attn_vs_matched`, etc.)
A non-parametric paired test on the per-item differences (e.g. `delta_prob_yes` for condition A minus `delta_prob_yes` for condition B, same items in both). Used instead of a paired t-test because the shift distributions aren't assumed normal. **p < 0.05** conventionally read as "the paired difference is unlikely to be pure chance." Caveat repeated throughout the reports: the inserted phrase is *identical* across all items in a condition, so items aren't independent draws of "the phrase's effect", meaning these p-values (often 1e-15 or smaller) understate the real uncertainty and shouldn't be quoted as if they were.

- `p_delta_vs_zero`: is the condition's shift different from no shift at all?
- `p_delta_vs_legacy_placebo`: is this condition's shift different from the original single tea-sentence placebo's shift, same items, paired?
- `p_delta_vs_matched` / `p_attn_vs_matched`: same test but against that condition's exact token-matched placebo instead.

### Spearman rank correlation (`spearman_entropy_vs_delta`, and the ad-hoc group-alignment checks)
Rank correlation, range -1 to +1, robust to non-linear monotonic relationships and outliers (unlike Pearson). Used for:
- **Model tracks human average**: `prob_yes` vs `p_offensive` across items (0.65, both models pooled loosely track the human rate).
- **Does disagreement predict sensitivity**: item's human `entropy` vs that item's `delta_prob_yes` under real framings (weak and inconsistent: -0.11 in Llama, +0.14 in Qwen).
- **Does the shift track real group opinion**: item's group rating gap vs that item's `delta_prob_yes` under the matching framing (near zero, 1 of 20 tests significant, this is the main negative finding motivating steering).

### `roc_auc`, `accuracy`, `precision`, `recall`, `f1`, `f1_macro` (zero-shot classification phase only, `zeroshot_two_datasets.py`)
Standard classification metrics (via `sklearn.metrics`), computed once per model/dataset on the 500-item zero-shot evaluation, not part of the attention-probing pipeline. `roc_auc` uses the continuous `prob_one` score and is threshold-independent, so it's identical whether reported under the greedy-decoding rule or the F1-tuned threshold rule for the same model/dataset. These describe the model's raw classification quality against the human majority label, from the earlier benchmarking phase, and are separate from everything in sections 2-5 above, which is about *how framing changes* the prediction, not whether the prediction is "correct."

---

## 7. Quick interpretation cheat sheet

| You see... | It means... |
|---|---|
| `delta_prob_yes` positive, large | framing pushed the model toward "offensive" |
| `delta_prob_yes` negative | framing pushed toward "not offensive" |
| `mean_abs_delta` similar across two conditions but `mean_signed_delta` opposite sign | two real, opposite effects that look "equal" only if you drop the sign, don't do that |
| `pct_demographic_norm` higher for condition A than B | the model's attention (post-sink) allocates more to A's phrase than B's, not proof A "matters" more behaviourally |
| `delta_entropy` negative | framing made the model more confident (less torn) than it was unframed |
| `delta_entropy` positive | framing made the model less confident |
| `flip_rate` low but `mean_signed_delta` large | the framing moves probabilities a lot but rarely enough to cross the 0.5 decision boundary |
| Wilcoxon p very small (e.g. 1e-15) on a per-item comparison within one condition | expected and somewhat inflated, since the phrase is identical across items; don't read it as extraordinarily strong evidence |
| Spearman near 0 between shift and real group gap | the model's behavioural change doesn't correspond to what that real group actually thinks, the core steering-motivation finding |
