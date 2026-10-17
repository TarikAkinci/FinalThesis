# Status update: v2 results and the plan for steering

## What v2 is

- Two models (Llama-3.1-8B, Qwen2.5-7B), each given the same 500 D3CODE texts.
- Every text is asked about once without an identity, and then with 22 different identity phrases: 14 real identities and 8 neutral placebo phrases such as "You are a runner.".
- Each phrase is tested in 10 different prompt formats (where the identity sits, how it is worded).
- The texts are ones both models are unsure about, so the identity has room to change the answer.

## Main findings

1. **Region is the only identity that really matters.** Naming a region ("You are someone from Sub-Saharan Africa") pushes both models toward "offensive" much more than a neutral phrase of the same length does. This holds in both models and all 10 prompt formats.
2. **Gender, religion and "an Arab / a European" behave like neutral phrases.** "You are a woman" has about the same effect as "You are a runner". The exception: Llama starts using gender when the prompt adds that background shapes how you read things.
3. **Qwen reacts to almost any persona.** Even "You are a cyclist" raises its average P(offensive) from 0.53 to 0.85.
4. **The wording barely matters; the position does.** Four different wordings ("You are X", "Imagine you are X", ...) give the same results. Putting the identity after the question pulls all answers down, and in Qwen it reverses the identity effect.
5. **Framing moves the models away from real people, not toward them.** We compared each answer with what raters from that region or gender actually said:
   - average distance grows from 0.35 to 0.51 (Llama) and from 0.40 to 0.68 (Qwen);
   - there is no item-level link to the real group differences (0 of 240 tests significant).
6. **Contested texts are not more sensitive.** How much human raters disagree does not predict how much the identity changes the model's answer.

## What changed compared with v1

- **Same texts for both models.** v1 used 236 texts for Llama and 64 for Qwen. v2 uses the same 500 for both, drawn from all of D3CODE.
- **Every identity in every prompt format**, instead of separate blocks with different conditions.
- **Better placebos.** Each identity is compared with neutral phrases of exactly the same length, identical in both models. The placebo in the context prompts no longer contains the word "offensive".
- **Stricter statistics.** The phrase is the unit of analysis, with phrase length controlled. v1's item-level p-values were too optimistic.
- **Technical fixes:** more precise answer scores, and two bugs in locating the phrase in the prompt.
- **Conclusions that changed:**
  - v1 said all real identities beat the placebo; v2 shows that is mostly region.
  - v1's correlation of +0.6 with real group differences does not hold up in v2.

## How we started with steering

- **Which attention heads react to region names?** Looking inside the models: specific heads, mostly in the middle layers, pay more attention to region phrases than to neutral phrases of the same length. That is 34 heads in Llama (most at layer 14, consistent with v1) and 108 in Qwen. In some of them, more attention goes with a stronger push toward "offensive".
- **Running now:**
  - Control phrases with the same place name but no identity ("someone researching Western Europe"). They separate "reacts to an identity" from "reacts to a place name".
  - A test that turns up each head on its own and measures how much the answer changes (PASTA's head-selection method).

## Proposed direction

1. **Causal test.** On texts not used for selection, switch the found heads off or turn them up, and compare with randomly chosen heads. Do they really carry the region effect?
2. **Steering**, with 4 regions (Sub-Saharan Africa, Arab Culture, Western Europe, Latin America) and 3 prompt formats:
   - amplify attention to the region (does more attention mean more effect?);
   - try to create an effect for gender and religion, where prompting alone has none;
   - try to move the model toward the real group's ratings (the main goal);
   - steer attention to the text itself instead of the identity, to see whether that reduces over-flagging.
3. **Comparisons:** prompting alone, random heads, neutral phrases, and a second steering method that does not use attention (activation steering).

## Questions for you

- Is the focus on region, with gender and religion as "can steering create an effect?" cases, the right direction?
- Is "closer to the real group's ratings on unseen texts" the right success criterion for steering?
