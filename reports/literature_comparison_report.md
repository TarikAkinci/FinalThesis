# How Our Five Findings Compare to the Literature

I went through the six papers your supervisor gave you (summarized in `technical_papers_sum.md` and `attention_probing_papers_summary.md`) and searched for the papers most directly relevant to each of your five findings, since none of the six is actually about demographic-framing effects on toxicity/offensiveness classification specifically, they're mostly about political-ideology steering and general fairness debiasing. Here's how each finding lines up.

## Finding 1: Demographic framing reliably shifts the model toward "offensive," and this survives a placebo control

This is a persona-conditioning effect, and it sits inside an active, well-established line of work on exactly that.

**Aligns with:**
- **Salewski et al., 2023, "In-Context Impersonation Reveals Large Language Models' Strengths and Biases"** (NeurIPS 2023 spotlight). They show that simply telling a model to adopt a persona changes its downstream output in content-relevant ways, e.g. a model told it's "a man" describes cars better than one told it's "a woman." That's the same basic mechanism as ours: a one-sentence identity statement, nothing else changed, produces a measurable, systematic shift in behavior on an unrelated task.
- **Gupta et al., 2024, "Bias Runs Deep: Implicit Reasoning Biases in Persona-Assigned LLMs"** (ICLR 2024). Across 24 reasoning datasets and 19 personas, they find persona assignment causes systematic behavioral shifts (reasoning errors, abstentions) even though the same models explicitly reject stereotypes when asked directly, bias appears "underneath a veneer of fairness," in their words. That framing fits your result well: nothing in our prompt asks the model to be biased, or even mentions offensiveness policy, the identity clause alone does the work.

**What our placebo control adds that these two don't do directly:** neither paper rules out "any persona, regardless of content, does this." Your placebo condition is closer to **Zhao et al., 2021, "Calibrate Before Use"**, who established using a *content-free* input (they use the literal string "N/A") as a calibration baseline to isolate a model's prompt-format-driven bias from its content-driven bias. Your placebo is the same idea applied to a different question: not "what's the model's default answer distribution" but "does an irrelevant sentence move the decision the same way a demographic one does." That you found it does *not* move it the same way, in fact it moves it in the *opposite* direction, is a stronger and more specific result than either Salewski or Gupta report, because you have the control condition they don't use.

## Finding 2: "Arab Culture" produces a larger effect than "Western Europe," "man," or "woman"

This is the finding with the most direct precedent in the literature, and it's a good one to lean on.

**Aligns closely with:**
- **Abid, Farooqi & Zou, 2021, "Persistent Anti-Muslim Bias in Large Language Models"** (AAAI/ACM AIES). Using GPT-3, they found "Muslim" gets analogized to "terrorist" in 23% of completions, far more than the equivalent association for any other religion tested (e.g. "Jewish" to "money" at 5%). That's the same qualitative shape as your result: one specific identity category draws a disproportionately strong association with a negative concept (violence there, offensiveness here), while comparison categories tested the same way don't show it nearly as strongly.
- This also matches your own earlier religion_muslim vs. religion_christian result from the main attention-probing report, where Muslim drew ~27% more relative attention than Christian, so you actually have two independent axes (religion in the earlier work, region here) both pointing at the same asymmetry, which is worth stating together.
- **BBQ** (Parrish et al., 2022) and **CrowS-Pairs** (Nangia et al., 2020) both include religion and nationality/ethnicity as explicit bias axes and consistently find Muslim- and Middle-Eastern-coded categories among the most stereotyped across many models, not a one-off artifact of one paper or one model family.

**What's new here:** Abid et al. tested GPT-3 via text completion in 2021. You're finding the same qualitative asymmetry in Llama-3.1-8B-Instruct in 2026, via a completely different method (attention share and calibrated probability shift, not completion sampling). That's a genuine, worth-stating contribution: the same category of bias, persisting across model generations and measurement methods, rather than being an artifact of one specific technique.

## Finding 3: The placebo pushes the opposite direction, toward "not offensive"

This is the finding with the thinnest direct precedent, worth being honest about that in the thesis rather than overclaiming a match.

**Related, not a direct match:**
- Gupta et al.'s "veneer of fairness" framing is the closest conceptual cousin: it says bias is present even when models "overtly reject stereotypes," implying the model's stated/default behavior is decoupled from what actually drives its outputs under persona framing. Your finding is a specific instance of that decoupling: a *non-identity* persona also decouples the model from its unconditioned behavior, just in the other direction.
- The general safety-training literature (e.g. work on why safety behaviors are brittle to surface-level prompt changes, such as Wei et al.'s work on jailbreak mechanisms) supports the general claim that "safety-adjacent" classification behavior in these models is more sensitive to surface prompt framing than to deep task understanding, consistent with a content-free sentence measurably loosening the model's offensiveness threshold.
- I did not find a paper that specifically tests "does an irrelevant persona reduce classification vigilance relative to baseline." That looks like a genuinely open, specific question your placebo result speaks to directly. Frame it as a hypothesis the data supports, not as confirming prior work, that's the honest and, frankly, more interesting way to present it: a testable mechanism (identity-framing raises caution, personal-preference framing lowers it) that the existing literature hasn't isolated this cleanly.

## Finding 4: Naive entropy-based disagreement selection has a baseline-ceiling confound

This finding sits at the intersection of two literatures that don't talk to each other much, which is exactly why it's worth keeping.

- **Uma et al., 2021, "Learning from Disagreement: A Survey"** (JAIR) and **Plank, 2022, "The 'Problem' of Human Label Variation"** (EMNLP) establish the field your item-selection method belongs to: treating annotator disagreement as signal, not noise, and modeling it directly rather than collapsing to a majority vote. Your entropy-based selection is a straightforward, correct application of that framework.
- Neither paper, nor the six your supervisor gave you, addresses what happens when you select items by *human* disagreement and then measure how a *model's own* prediction moves. That combination, human-disagreement-driven item selection feeding into model-behavior measurement, is less common than either half alone (disagreement modeling on its own, or bias benchmarks like BBQ/CrowS-Pairs that use hand-picked rather than disagreement-selected items). The ceiling-effect confound you found (entropy-selected items are ones the model already calls "offensive" with near-certainty, since that's what generates human disagreement in the first place) is a real, specific, and as far as I can tell undocumented pitfall of doing exactly that combination. It's a legitimate, if modest, methodological contribution: worth a paragraph in your methods section framed exactly that way, "this is a confound anyone combining these two approaches should check for."

## Finding 5: Prompt position matters, and not in a simple "more salient = bigger effect" way

This one has the strongest, most direct grounding of all five.

**Aligns closely with:**
- **Zhao et al., 2021, "Calibrate Before Use"** names the exact mechanism you found: **recency bias**, language models systematically over-weight whatever sits near the end of the prompt, independent of its content. That's precisely what your `suffix` result shows, everything (demographic or placebo) collapses to a uniform "not offensive" push once it's the last thing before generation, which is a content-independent positional artifact, not a content effect being amplified.
- **Liu et al., 2023, "Lost in the Middle: How Language Models Use Long Contexts"** establishes the broader point your finding is a special case of: position within a prompt affects how much a model uses information in a non-monotonic way (their result is U-shaped across a long context; yours is a sharp discontinuity between "early/mid-prompt" and "the last thing before the model has to answer"). Different task, same underlying phenomenon: position is not a neutral variable you can ignore or treat as simply correlating with salience.

**What this gives you:** a citable, well-established mechanism (recency bias) to explain *why* `suffix` broke, rather than presenting it as an unexplained anomaly. It also retroactively justifies your methodological choice to run the rest of this project's probing on the `prefix` structure: per Zhao et al.'s own finding, prefix position is comparatively less contaminated by recency effects than suffix, so your main results were built on the more defensible structural choice from the start, this check confirms that after the fact rather than revealing a problem with earlier work.

---

## One-paragraph summary for your supervisor

Findings 1 and 2 replicate, on a current instruction-tuned model with a placebo-controlled attention/probability method, patterns already documented in the persona-bias literature (Salewski et al., Gupta et al.) and, for the specific Muslim/Arab-culture asymmetry, in bias-in-completion work going back to GPT-3 (Abid et al., 2021). Finding 5 has a named mechanism in the calibration literature (Zhao et al.'s recency bias) and a structural cousin in long-context research (Liu et al.'s "lost in the middle"). Findings 3 and 4 are the more original contributions: finding 3 isolates a specific, previously-untested asymmetry (identity framing raises apparent caution, irrelevant framing lowers it) that the existing "veneer of fairness" framing predicts the *shape* of but hasn't shown directly; finding 4 is a concrete methodological pitfall at the intersection of the disagreement-modeling and bias-probing literatures that neither literature currently flags on its own.

## Sources

- [In-Context Impersonation Reveals Large Language Models' Strengths and Biases (Salewski et al., NeurIPS 2023)](https://arxiv.org/abs/2305.14930)
- [Bias Runs Deep: Implicit Reasoning Biases in Persona-Assigned LLMs (Gupta et al., ICLR 2024)](https://arxiv.org/abs/2311.04892)
- [Calibrate Before Use: Improving Few-Shot Performance of Language Models (Zhao et al., 2021)](https://arxiv.org/abs/2102.09690)
- [Persistent Anti-Muslim Bias in Large Language Models (Abid, Farooqi & Zou, AIES 2021)](https://www.researchgate.net/publication/353604819_Persistent_Anti-Muslim_Bias_in_Large_Language_Models)
- [BBQ: A Hand-Built Bias Benchmark for Question Answering (Parrish et al., 2022)](https://arxiv.org/abs/2110.08193)
- [Learning from Disagreement: A Survey (Uma et al., JAIR 2021)](https://www.jair.org/index.php/jair/article/download/12752/26751/29240)
- [The "Problem" of Human Label Variation (Plank, EMNLP 2022)](https://aclanthology.org/2022.emnlp-main.731/)
- [Lost in the Middle: How Language Models Use Long Contexts (Liu et al., 2023)](https://arxiv.org/abs/2307.03172)
