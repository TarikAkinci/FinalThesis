# Attention Across Layers: Where the Sink Lives, and Where the Demographic Effect Lives

Method (for the record): Llama-3.1-8B-Instruct, single deterministic forward pass per item/condition (no `.generate()`, no sampling anywhere — equivalent to temperature 0), `output_attentions=True`, attention read at the last token position (the position that produces the 0/1 decision), averaged over the 32 heads **within** each layer but *not* collapsed across layers — giving one number per (item, condition, layer). 30 items, stratified across D3CODE's 3 categories, each run under all 7 conditions.

## 1. The attention sink is not uniform across depth — it has a clear W-shape

| Layer band | Sink share (`pct_other_scaffolding`) | Pattern |
|---|---|---|
| 0 | 58.5% | Low — the model hasn't yet consolidated onto the sink |
| 1–6 | 81–93% (peak at layer 2: **93.3%**) | Sink forms almost immediately and dominates |
| 7–14 | 55–77%, bottoming at **layer 14: 53.5%** and layer 13: 54.8% | **The trough** — the layer band doing the most genuine content processing |
| 15–20 | 69–82%, climbing back up | Sink re-forms |
| 21–31 | 79–95%, peak at **layer 25: 95.1%** | Sink dominates almost completely near the output |

This is consistent with the broader attention-sink literature (Xiao et al., 2023): sinks form very early (by layer 1–2) as a structural, largely content-independent phenomenon, and the layers doing meaningful semantic work sit in a mid-network band, sandwiched between an early "settling" phase and a late "consolidation toward output" phase where the model again leans heavily on fixed anchor tokens. The precise trough we found — **layers 13–14** — lines up closely with the specific layers your supervisor's political-ideology papers flagged as most informative for Llama-3.1-8B-instruct (layers 11, 13, 26–31 in the multilingual political-steering paper), even though that paper probes attention-*head value outputs* rather than raw attention scores. That's a different signal, so treat it as a suggestive cross-check, not proof of the same mechanism — but it's a reasonable point to make out loud in a presentation.

## 2. The demographic-attention asymmetry (Muslim vs. Christian) is concentrated in exactly that trough

Raw per-layer demographic attention share, `religion_muslim` vs `religion_christian`, with the relative gap:

| Layer band | Muslim vs Christian | Relative gap |
|---|---|---|
| 0–8 | close, direction inconsistent | −15% to +19% (noisy, near parity) |
| **9–18** | Muslim consistently and substantially higher | **+57% to +170%**, peaking at **layer 13 (+170%)**, also layer 16 (+101%), layer 14 (+92%), layer 17 (+85%), layer 11 (+83%) |
| 19–31 | Muslim still higher in nearly every layer, but gaps shrink in absolute terms as the sink reclaims most of the attention budget | +5% to +93%, noisier |

The single largest relative gap (**+170% at layer 13**) occurs at almost exactly the layer with the lowest sink (54.8%) — i.e., the layer with the most attention "available" for content. That's not a coincidence to wave away as noise; it's the expected shape if the demographic-attention effect is a genuinely semantic phenomenon rather than a numerical artifact: it should show up most clearly where the sink isn't already eating the attention budget.

**What we expected** going into this: either a uniform effect across all 32 layers, or an effect confined to a single "decision" layer near the very end (close to the classification head). **What we actually see** is neither — a broad, consistently-signed effect (Muslim > Christian in 28 of 32 layers) with its magnitude sharply concentrated in a specific mid-network band (roughly layers 9–18), overlapping with — but not identical to — the layers the political-ideology literature already flagged as informative for this exact model.

## 3. Two things to fix before this goes in front of your supervisor as a finished claim

1. **The raw per-layer numbers above are still sink-confounded.** A layer with 95% sink (layer 25) mechanically has less room left for the demographic phrase than a layer with 55% sink (layer 14), so part of the "shrinking gap in later layers" story could just be the sink squeezing the numbers, not a real drop in relative salience. The fix is the same one already applied at the aggregate level — renormalize `pct_demographic` by `(1 − pct_other_scaffolding)` **per layer** — and it's free to compute from data already on the cluster.
2. **No significance test per layer yet.** "Layer 13 shows +170%" is a mean over the same 30 paired items used everywhere else; before calling it out by name in a presentation it needs the same treatment as the aggregate asymmetry — a paired Wilcoxon signed-rank test per layer, to confirm the trough band is where the effect is *statistically*, not just numerically, strongest.

I've written the script for both (`analyze_attention_layerwise.py`) — it reads `attention_probe_layerwise.csv`, which is already sitting on the cluster from the last run, so this needs no GPU time, just a couple of minutes on the login node.
