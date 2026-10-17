# What we found so far, explained from the ground up

This note explains the v2 run ("run A") and the first v3 analysis step (the per-head analysis) without assuming background in attention or steering. Section 1 is the minimum theory. Sections 2 to 6 walk through what we measured and found, with one concrete example head. Section 7 says what it means for run B.

## 1. The minimum theory

**Tokens.** The model never sees words, it sees tokens: pieces of words. "You are someone from Sub Saharan Africa." is about 9 tokens. The whole prompt (chat template, identity sentence, item text, question) is about 100 to 250 tokens.

**Layers.** The model processes the prompt in a stack of layers: 32 in Llama, 28 in Qwen. Each layer updates its internal representation of every token a little, using information from the tokens before it. Early layers mostly handle word-level things. Middle layers combine meaning. Late layers prepare the output.

**Attention and heads.** Attention is the mechanism by which a token gathers information from earlier tokens. Each layer has many attention heads working in parallel: 32 per layer in Llama (1,024 in total), 28 per layer in Qwen (784 in total). A head is like one reader with one question in mind. For every position it decides how much to look at each earlier token, and those weights always add up to 100%. Different heads specialise. Some track the previous word, some find the subject of a sentence, some look at the instruction.

**The answer position.** The model's answer ("0" or "1") is produced at the very last position of the prompt. So the question "what does the model look at when it decides?" becomes "where do the heads at the last position put their attention?" That is what we recorded: for every prompt, every layer and every head, the share of attention from the last position to four parts of the prompt:
- the identity sentence ("You are someone from ...");
- the instruction ("Is this text offensive? Answer with only 0 or 1.");
- the item text;
- everything else (the chat template and the first token).

**The attention sink.** Many heads put most of their attention on the very first token of the prompt, which carries almost no information. This is a known phenomenon (Xiao et al.). It works like a parking spot for attention when a head has nothing useful to do. In v1 about 80% of all attention went there.

**Raw vs value-weighted attention.** Raw attention says where a head *looks*. But a token can be looked at a lot and still pass on very little, because the information it hands over (its "value") is tiny. The sink token is the classic case. Value-weighted attention (Kobayashi et al.) multiplies each look by how much the token actually passes on. It is closer to "what flows into the decision". We stored both.

**Steering (PASTA).** Steering means changing the attention during the forward pass, without retraining. PASTA picks a set of heads and forces them to look more at a chosen span: it scales attention to everything else by 0.01 and renormalises. If those heads carry the identity information, turning them up should change the answer in a predictable way. So steering is the causal test, and later the tool. The per-head analysis exists to find the heads worth steering.

## 2. What we recorded in run A (v2)

- 500 items, the same for both models, each under 22 conditions (14 identities, 8 content-free placebos such as "You are a runner.") and 10 prompt variants. That is 110,500 prompts per model.
- For each prompt: the answer (P(offensive) and log-odds), plus the attention shares described above for every one of the 1,024 (Llama) or 784 (Qwen) heads, raw and value-weighted. That is the 1.6 GB and 1.2 GB per-head file on your laptop.
- The behavioural findings from these prompts are in `v2_behavioural_analysis.md`. The one that matters here: region phrases change the answer more than content-free phrases of the same length, in both models and every variant. Gender, religion and the bare identity words do not. That is why v3 focuses on regions.

## 3. Step 1: which heads treat the region phrase specially?

We used only half of the items (split A, 250 items). The other half is untouched for the causal test in run B, so we never test heads on the same items we picked them on.

### 3.1 Throwing out the sink heads

A head that puts more than 95% of its attention on the sink and the template is not really reading the content. Its attention to the identity phrase is tiny and noisy. In Llama that is 302 of 1,024 heads, in Qwen 104 of 784. They are excluded from the search.

### 3.2 The length problem, and how the content test solves it

A longer phrase gets more attention simply because it has more tokens to look at. A region phrase like "someone from the Arab Culture region" is longer than "a runner". So "this head gives regions more attention" can just mean "regions are longer". In the data, a head's attention to a phrase follows the phrase's length with a typical rank correlation of about 0.5 across conditions.

The fix has two parts:
1. Use attention **per token** of the phrase instead of the total.
2. Compare each region with what a **content-free phrase of the same length** would get. With 8 placebos at 4 different lengths, we fit a straight line "per-token attention vs length" through the placebos for every head. Then we ask how far above or below that line the region phrases sit. If they sit far outside the scatter of the placebos around their line, the head treats the region *content* specially.

**Example: Llama, layer 19, head 1** (prefix family, per-token attention × 1000):

| phrase | length (tokens) | attention | placebo line at that length |
|---|---|---|---|
| "a cyclist" / "a runner" | 6.2 | 1.32 / 1.18 | 1.23 |
| "an early riser" / "someone who plays chess" | 8.2 | 0.89 / 1.15 | 1.06 |
| "someone who collects old maps" / "... crossword puzzles" | 9.2 | 0.90 / 1.04 | 0.97 |
| "... takes the stairs" / "... tea over coffee" | 10.2 | 0.92 / 0.90 | 0.89 |
| **Western Europe** | 8.2 | **1.19** | 1.06 |
| **Latin America** | 8.2 | **1.54** | 1.06 |
| **Arab Culture** | 10.2 | **1.41** | 0.89 |
| **Sub-Saharan Africa** | 10.2 | **1.67** | 0.89 |

The placebos hug their line closely. The regions sit 10% to 90% above it. Pooled over the four regions, that gives t = 7.7. The absolute numbers are small (about 0.1% of the head's attention per token) because this head spreads attention over the whole prompt. What matters is the comparison.

Counts (raw attention, non-sink heads, false discovery rate controlled at 5%): in the prefix family, 152 Llama heads and 282 Qwen heads give regions more (or less) attention than content-free phrases of the same length would get. Almost all of them give *more*.

### 3.3 The behaviour link, and the confound we caught

Paying special attention is not enough; it should matter for the answer. So for every item we compute:
- **dA:** how much more attention the region gets than its two same-length placebos, in that head;
- **dB:** how much more the region shifts the answer (in log-odds) than those placebos.

If a head carries the region information into the decision, items where it attends more to the region should be items where the region moves the answer more. That is a correlation across the 250 items, per head.

The first version of this said that 468 of Llama's heads were linked to the answer, far too many. The reason is **text length**:
- longer item texts give every phrase a smaller slice of attention (attention is shared, and there are more tokens);
- longer texts also make the region effect smaller (rho = -0.41).

So nearly every head looked "linked" because both sides depended on text length. The fix is a partial correlation: remove text length and the item's baseline answer from both sides first, then correlate what is left. That brought Llama's prefix-family count from 468 down to 94.

For the example head, the partial correlation is 0.65: on items where this head gives the region more attention, the region pushes the answer more.

### 3.4 Stability

We split the 250 items into two random halves and repeated everything.
- The content test barely changes between halves (agreement 0.996). That sounds great but is expected: both halves use the same phrases, and the content test is about phrases. It is not independent evidence.
- The behaviour link is the honest stability check, because it lives at the item level. Across heads it agrees at 0.32 between halves for Llama and 0.64 for Qwen. Llama's links are noticeably fragile. The example head is stable (0.67 and 0.65 in the two halves).

### 3.5 The frozen candidate rule

A head is a candidate only if all of these hold:
- it is not a sink head;
- the content test is significant, with the same direction, in at least 2 of the 3 structures (prefix family, infix, target);
- the behaviour link is significant in at least 1 structure;
- the content effect and the direction of the behaviour link hold in both halves of the data.

The rule was written to a file before any causal test.

## 4. Results of step 1

| | Llama | Qwen |
|---|---|---|
| heads | 1,024 | 784 |
| sink heads (excluded) | 302 | 104 |
| candidates, raw attention | 34 | 108 |
| candidates, value-weighted | 40 | 123 |
| overlap raw / value-weighted | 21 | 86 |
| where | layers 8 to 24, most at layer 14 | spread over layers 7 to 24 |

What this means:
- **Specific heads give region names special attention, mostly in the middle layers.** For Llama this lines up with v1, where the head-averaged effect sat in layers 9 to 18 and peaked at layer 13. Middle layers are where models typically combine meaning, so this is plausible.
- **Almost all candidates give regions *more* attention, never less.**
- **The link to the answer goes in both directions.** For most candidates (about 60% in Llama, 73% in Qwen), more attention to the region goes with a bigger push toward "offensive". For the rest it goes with a smaller push. So these heads are not all "pushers". Some may dampen the effect. Run B will tell which is which, because switching a head off shows its real role.
- **The three structures overlap but differ.** Across heads, the content test agrees at about 0.78 between the prefix family and infix, but only about 0.65 between either of them and target. Target ("offensive *for* X?") uses partly different heads. That fits the behavioural finding that target is a different kind of question (is the text about the group?) rather than a perspective.
- **Raw and value-weighted attention mostly agree.** Llama's two candidate lists share 21 heads, Qwen's 86. Value-weighting removes the sink's dominance (it has far fewer "sink heads"), but it identifies largely the same content heads.

## 5. What this does NOT show yet

- **It is correlational.** A head that looks at the region phrase may just be along for the ride, while the real work happens elsewhere. Only an intervention can show that a head matters. That is run B.
- **"Region" vs "capitalised place name".** All 8 placebos are lowercase everyday words. Region names are capitalised place names. A head that simply likes proper nouns or geography would pass the content test without "knowing" anything about identity. That is why run B starts with the place-name controls: the same region names in phrases that say nothing about belonging ("someone researching Western Europe", "someone mapping Western Europe"). If a head gives "someone from Western Europe" more attention than "someone researching Western Europe", with exactly the same place tokens and the same length, that is attention to the *identity*, not to the place name.
- **Llama's behaviour links are fragile** (0.32 between halves), so its candidate list may shrink once we test it causally.

## 6. One-paragraph summary

When the models decide whether a text is offensive, a few dozen attention heads (Llama) or about a hundred (Qwen), mostly in the middle layers, look at a region phrase more than at a content-free phrase of the same length. In some of these heads, the extra attention goes with a bigger push toward "offensive". This was found after removing two traps that would otherwise have produced false hits: phrase length and item-text length. It is still correlational, and it cannot yet separate identity from place names. Run B fixes both.

## 7. What run B does with this

1. **Place-name controls (B1).** Rerun the 500 items with "someone researching / mapping <region>" for the 4 regions, then redo the content test against these. Candidates that only like place names drop out. The candidate list is then frozen for good.
2. **PASTA profiling (B2).** Steer each head on its own toward the region phrase and measure how much the answer changes beyond placebos and place controls. That ranks heads by causal effect, independently of step 1.
3. **Ablation (B3, after the freeze).** On the held-out items, switch the top candidates off (attention to the region phrase set to zero) or turn them up (×2, ×4), and compare with 10 random sets of heads from the same layers. If the candidates change the region effect more than random heads do, they really carry it. Two upper bounds tell us how much of the effect runs through attention to the phrase at all.
