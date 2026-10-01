# Disagreement + Placebo Study, v3: Seeded-500 Sample, Entropy-Selected Items

Both studies reran on the new item pool your supervisor asked for: 100 items selected from *within* the seeded 500-item D3CODE evaluation sample (the same one the headline zero-shot metrics are reported on), ranked by Shannon entropy of each item's full rating distribution (reusing `evaluate.py`'s own entropy function, not a fresh implementation). No axis partitioning this time, every item gets all 6 conditions. The `reworded` bug (length-asymmetric phrasing) is fixed. Both ran cleanly, zero span-finding failures across 2700 total rows.

## First, a confound I need to flag before anything else

I checked the baseline predictions on this new item set, and it's heavily ceiling-concentrated: 95 of 100 items are already predicted "offensive" at baseline, with a median confidence of P(yes)=0.999. That's more extreme than the earlier gap-based sample (86.6% offensive, median 0.989).

This makes sense once you think about why: an item with maximum rating entropy is one where the human population is split roughly 50/50, and that tends to happen on content that has *some* real edge to it, not neutral text (genuinely inoffensive text gets near-unanimous "not offensive" ratings, which is *low* entropy). So selecting for high population-level disagreement tends to select for content the model itself reads as confidently offensive, even though humans are internally divided about it. That's a real and interesting fact on its own, but it also means there's very little room left for any prompt perturbation, demographic or placebo, to push the prediction *further* toward "offensive." Almost all the room to move is downward. So before reading anything below: expect almost every condition to show a net-negative average shift, and don't read that alone as evidence about demographic content specifically. It's a property of this item selection, not of what's being tested.

## Finding 1: the attention-share result is now very robust

Across all 4 prompt structures (prefix, suffix, reworded, embedded), the same pattern holds without exception:

| condition | wins or loses vs. placebo, all 4 variants |
|---|---|
| gender_man | loses, every variant, p < 4e-16 |
| gender_woman | loses in 3 variants (p < 1e-16); a coin flip at `suffix` specifically (p=0.12) |
| region_western_europe | loses, every variant, p < 4e-9 |
| region_arab_culture | **wins, every single variant**, p < 4e-15, strongest at `suffix` (+0.030) |

That's the cleanest possible replication: 4 independent prompt structures, same qualitative answer every time for 3 of 4 conditions, and Arab Culture is the one consistent exception in every structure tested, not just the original one.

## Finding 2: the reworded fix worked

With the length-asymmetry bug fixed, `reworded` no longer looks anomalous. It now matches `prefix` closely on every condition (e.g. region_arab_culture: prefix +0.0185, reworded +0.0226, both clearly significant, both clearly smaller than `suffix`'s +0.0296). That's a good sign the fix addressed the actual problem rather than just changing the numbers arbitrarily. I'd treat the original "reworded flips the story" result as fully retracted now.

## Finding 3: on raw behavioral magnitude, placebo wins almost everywhere now, including Arab Culture, except at one specific position

| condition | prefix | suffix | reworded | embedded |
|---|---|---|---|---|
| gender_man | loses (p=7e-13) | loses (p=1e-8) | loses (p=1e-14) | loses (p=3e-17) |
| gender_woman | loses (p=3e-13) | loses (p=0.028) | loses (p=3e-15) | loses (p=2e-16) |
| region_western_europe | loses (p=3e-10) | tied (p=0.31) | loses (p=1e-12) | loses (p=2e-17) |
| region_arab_culture | **loses (p=1e-6)** | **wins (p=0.0002)** | loses (p=1e-9) | loses (p=9e-15) |

This is different from the earlier n=30 check, where Arab Culture wasn't significant either way on |ΔP(yes)| at `prefix`. At n=100 it's now significant and *loses*. The one place Arab Culture wins on both attention share and actual probability movement is `suffix`, the position closest to generation. That's a real, specific, and now-replicated pattern worth naming directly: Arab Culture's behavioral signal shows up most clearly when it's the last thing before the model has to answer, not when it's buried earlier in the prompt.

## Finding 4: even inside the ceiling effect, demographic and placebo are not the same shape

Given finding 3 could just be "everything loses to placebo because of the ceiling effect," I checked direction, not just magnitude, at `prefix`:

| condition | items pushed negative | items pushed positive | mean signed Δ |
|---|---|---|---|
| gender_man | 63 | 33 | +0.009 |
| gender_woman | 80 | 20 | +0.001 |
| region_western_europe | 65 | 32 | +0.014 |
| region_arab_culture | 76 | 24 | +0.011 |
| placebo | 91 | 6 | **-0.027** |

Every demographic condition still has a *positive* mean, despite most individual items moving negative, because a real minority of items get pushed substantially toward "offensive." Placebo has almost no such items (6 of 100) and a clearly negative mean. So the honest read isn't "demographic content is inert here too." It's: demographic framing retains a distinguishable, if now heavily ceiling-suppressed, capacity to push some items more offensive, while the placebo essentially only erodes confidence downward. That's a real qualitative difference the raw |Δ| comparison alone doesn't show.

## What I'd actually conclude right now

- The attention-share finding (gender + Western Europe ≈ or < placebo; Arab Culture > placebo) is now about as well-replicated as this kind of study gets: 2 independent item-selection methods, 4 independent prompt structures, always the same answer.
- The behavioral (P(yes)) comparison is genuinely confounded by this item set's baseline ceiling effect, and I'd be cautious presenting the raw magnitude numbers as clean evidence either way without that caveat attached. The signed/directional breakdown is the more trustworthy read, and it still shows demographic content behaving differently from placebo, just not as dramatically as the unsigned numbers alone suggest.
- The one clean, position-specific behavioral finding that survived scrutiny: Arab Culture shows a real edge over placebo, on both measures, specifically when positioned right before generation (`suffix`). That's worth stating as its own finding, not folded into the general "demographic vs. placebo" summary.

## Before we call this settled

If your supervisor wants a clean magnitude-based P(yes) comparison (not just directional), the ceiling effect needs addressing, most simply by checking whether the entropy-selection could also require baseline_prob_yes to be within some reasonable band (e.g. not already >0.98), so there's actually room on both sides for a condition to move it. I haven't done that since it changes what she explicitly asked for (rank by entropy/variance/disagreement rate) and I didn't want to unilaterally add a filter she didn't request. Worth raising with her directly before deciding whether it's needed.
