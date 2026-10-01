# Steps 3-4 in Detail: Baseline, Zero-Shot Llama, and Persona Ensembling

Before the three steps individually, one thing that needs to be decided **once and reused across all three** — otherwise your results won't be comparable to each other, which defeats the point of having a baseline at all.

## Step 0 — Shared setup (do this first, applies to 3, 4, and 5)

1. **Fix your train/dev/test split now, and never change it.** Split by item (not by rating), stratified by item category (moral/random/social-group) and ideally region, so all three sets have similar composition. A reasonable split: 70/15/15. Save the item IDs for each split to a file (`splits.json`) — every experiment from here on loads this same file rather than re-splitting, or your numbers across steps 3/4/5 won't be legitimately comparable.
2. **Decide your two prediction targets now**, since both matter for your thesis:
   - **Central tendency**: the mean/majority rating per item (standard "is it offensive" prediction)
   - **Disagreement**: some measure of spread per item — I'd use **entropy of the rating distribution** as your primary disagreement measure (handles the ordinal 0–4 scale better than raw variance, and is directly comparable to the entropy-based LLM metrics from later steps)
3. **Build one shared evaluation module now**, not per-step. A single `evaluate.py` with functions for MAE, RMSE, correlation (central tendency) and entropy/JSD/CE (disagreement) that every later step imports. This is worth the up-front time — it's what makes your final results table trustworthy (same metric implementation everywhere) and saves you rewriting it three times.
4. **Log every experiment run to a common results file** — even something as simple as appending a row to a `results.csv` (`experiment_name, split, metric, value, config_hash, timestamp`) as you go. You'll thank yourself when writing the results chapter and needing to remember exactly what config produced which number.

---

## Step 3 — Baseline (no LLM)

**Goal**: establish the floor. Every later result needs to beat this, or it's not adding value.

1. **Naive baseline first (always include this)**: predict the training-set mean rating for every test item, regardless of content. This sounds trivial but it's the number that makes any subsequent improvement legible — "our model achieves MAE 0.6 vs. a naive-mean MAE of 1.1" is a real sentence you'll want in your results chapter.
2. **Featurize the text.** Two options, both worth trying since you'll want to know if the difference matters:
   - Simple: TF-IDF vectors + a linear model (fast, interpretable, good sanity check)
   - Better: sentence embeddings (e.g., `sentence-transformers/all-mpnet-base-v2`) + gradient boosting (LightGBM/XGBoost) or a small MLP — this is the one likely to actually compete with your later LLM results
3. **Train two models on your Step 0 targets**: one regressor for mean rating, one for entropy/disagreement. These can literally be the same architecture with a different target column — no need to overengineer this.
4. **Also fit one model on item *metadata alone*** (category, region distribution of raters, text length) with no text features — this tells you how much signal comes from "what kind of item this is" vs. the actual text content, which is a genuinely interesting number for your discussion section (does disagreement come from content or from context/category?).
5. **Evaluate on your fixed test split** using the Step 0 evaluation module: MAE/RMSE/correlation for mean rating, correlation for entropy prediction.
6. **Formalize the output**: a single results table with rows = {naive mean, TF-IDF+LR, embeddings+GBM, metadata-only} and columns = {MAE, RMSE, correlation} × {mean-rating task, disagreement task}. Save this table (CSV + a rendered version) — this is a direct, reusable thesis table.
7. **Do a quick residual check before moving on**: plot prediction error against item category and region. If your baseline already fails much harder on specific categories (e.g., "social-group" items), note that now — it's a useful thing to track through Steps 4 and 5 to see whether LLM-based methods close that specific gap or not.

**Time estimate**: this is genuinely an afternoon-to-a-day task, no cluster needed — run it locally.

---

## Step 4 — Paradigm A: Zero-shot / few-shot Llama

**Goal**: does prompting a real LLM beat your Step 3 baseline on plain accuracy? This is necessary groundwork before Step 5, not just a nice-to-have — if Llama can't even get central tendency roughly right, persona ensembling on top of it won't be trustworthy either.

1. **Write your prompt template and freeze it.** Literally save it as a constant/config file, not inline in code you might tweak later without noticing — you need to be able to report the *exact* prompt in your methodology section.
   ```
   SYSTEM: You are annotating social media comments for offensiveness as part of
   a content-moderation research study. Rate each comment from 0 (not offensive)
   to 4 (extremely offensive).
   USER: Comment: "{text}"
   Respond with only a single digit from 0-4.
   ```
2. **Get the prediction via logits, not free-text generation.** You covered this in the white-box guide — instead of generating text and regex-parsing a digit (fragile, and burns tokens), do a single forward pass and read off the model's probability for each of the tokens `"0","1","2","3","4"` at the response position. This gives you a full probability distribution in one pass, and sidesteps parsing failures entirely.
   ```python
   import torch
   inputs = tok(prompt, return_tensors="pt").to(model.device)
   with torch.no_grad():
       logits = model(**inputs).logits[0, -1]  # logits for the next token
   label_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(5)]
   probs = torch.softmax(logits[label_ids], dim=0)
   predicted_rating = torch.arange(5).float() @ probs   # expected value, if you want a point estimate
   ```
3. **Run this over your fixed test split** (start with the test split only, not train/dev — you don't need Llama predictions on data you're not evaluating on for this step). Save every raw output: item_id, full probability vector, point-estimate rating, and the exact prompt used. This raw output table is valuable on its own — you'll reuse it in Step 5's evaluation and in later error analysis.
4. **Add a few-shot variant.** Same template, but prepend 3–5 labeled examples sampled from your *train* split (never test — that's leakage). Try to include at least one high-disagreement and one low-disagreement example, since that's directly relevant to what you're studying.
5. **Track refusals/malformed outputs as their own metric.** If the model ever declines to answer or the top predicted tokens aren't the digits you expect, log it rather than silently dropping the item — as flagged before, this can systematically bias your results if it's not random.
6. **Evaluate** using the same Step 0 module: MAE/correlation on mean rating, but now also break results out by **human disagreement level** (low vs. medium vs. high entropy items) — this is your direct test of the "overconfident judge" finding: does Llama's error rate spike specifically on the items where humans disagreed most?
7. **Formalize**: extend your Step 3 results table with rows for {zero-shot Llama, few-shot Llama}, same columns. Add one new figure: model error (or confidence/entropy) plotted against human disagreement level — this single plot is probably going into your results chapter regardless of what else you find.

**Time estimate**: a day or two once your cluster environment (from the previous guide) is confirmed working — this is your first real `sbatch` job.

---
