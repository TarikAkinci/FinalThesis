# Practical Guide: Recreating LLM-Based Annotation & Disagreement-Prediction Experiments

This is the "how do I actually build this" companion to your literature review. You already know ML architecture from coursework, so I'm skipping the basics of what a transformer is and focusing on what's specific to *this* line of work: using LLMs as simulated annotators, comparing them to real human disagreement, and modeling that disagreement directly.

---

## 1. The two experimental paradigms in this literature

Almost every paper you read does one (or both) of these two things. Knowing which one you're building matters because it changes your whole pipeline.

**Paradigm A — LLM-as-annotator.** You give the LLM a text and ask it to label it (offensive/not, or a Likert score), then compare that single prediction to the human gold label or majority vote. This is the simpler setup (Kamruzzaman, CreHate's zero-shot evaluation, the "overconfident judge" paper).

**Paradigm B — LLM-as-disagreement-predictor.** You're not asking "what's the right label," you're asking "what's the *distribution* of labels a diverse group of humans would give, or which two groups would disagree." This needs either (1) persona-conditioned prompting repeated across many simulated annotators, or (2) a model that outputs a distribution/soft label directly, evaluated against the real human distribution (Movva et al., Sarumi et al., Gordon et al.'s Jury Learning). **Given your thesis question, Paradigm B is the one you actually care about** — Paradigm A is really just a baseline you'll want to report to show why B is needed.

---

## 2. Getting set up: APIs, SDKs, keys, cost

For prompting experiments you don't need a GPU at all — everything runs through an API call. Three things to set up:

- **An API key** from whichever provider(s) you're testing (OpenAI for GPT-4/GPT-4o, Anthropic for Claude, or a provider like Together AI / OpenRouter if you want cheaper access to open models like Llama or Mixtral for comparison — several of your papers, e.g. Atil et al. and the ensembling paper, use multiple model families to show generality, which is a good practice to copy).
- **The provider's Python SDK** (`pip install openai`, `pip install anthropic`). Both use a similar pattern: instantiate a client, send a list of messages, get a response object back.
- **A cost estimate before you run anything at scale.** D3CODE has ~4,554 items. If you simulate even 20 personas per item across 2 models, that's ~180,000 calls. Do a cost estimate with a tiny model or cached test batch before scaling up — and check whether your provider offers a **batch API** (OpenAI and Anthropic both do), which is typically 50% cheaper for non-time-sensitive workloads like this, since you don't need results in real time.

```python
from openai import OpenAI
client = OpenAI(api_key="...")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are annotating social media comments for offensiveness."},
        {"role": "user", "content": "Rate this comment's offensiveness from 0 (not offensive) to 4 (extremely offensive): '<text>'"}
    ],
    temperature=0,   # deterministic — see section 5 for why this matters
)
print(response.choices[0].message.content)
```

A practical note: for offensive-content classification tasks, models sometimes refuse or hedge. Framing the system prompt clearly as a *content-moderation research task* (not a request to generate offensive content) reduces this substantially — several papers explicitly note this framing in their methodology sections, and it's worth doing the same in yours for reproducibility.

---

## 3. Zero-shot and few-shot annotation (Paradigm A, your baseline)

This is the simplest experiment: ask the model to label the item, nothing more.

- **Zero-shot**: just the instructions + the text. Fast to build, this is what most of your papers use as the baseline comparison point.
- **Few-shot**: include 2–5 labeled examples in the prompt before the target item. Useful for calibrating the model to D3CODE's specific 0–4 scale rather than its own internal notion of "offensive."

**Structured output matters a lot here.** Don't parse free text with regex — use the API's structured output feature so the model is constrained to return exactly the schema you need. OpenAI supports this via `response_format` with a JSON schema; Anthropic supports it via **tool use** (define a "tool" that's really just your desired output schema, and force the model to call it).

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "offensiveness_rating",
            "schema": {
                "type": "object",
                "properties": {
                    "rating": {"type": "integer", "minimum": 0, "maximum": 4},
                    "reasoning": {"type": "string"}
                },
                "required": ["rating"]
            }
        }
    }
)
```

This alone will save you a huge amount of debugging time versus regex-parsing free text, and it's what makes large-scale automated runs actually reliable.

---

## 4. Persona / sociodemographic prompting (the core technique for Paradigm B)

This is the method behind Sarumi et al., Kamruzzaman et al., Atil et al., and the "impact of annotator personas" paper. The idea: instead of asking the model to give *the* answer, you condition it on a specific annotator profile and ask what *that person* would say, then repeat across many profiles to build a simulated distribution.

```python
persona = {
    "age": 34, "gender": "woman", "country": "South Africa",
    "moral_values": "high Care, high Purity"  # if you're using D3CODE's MFQ data
}

system_prompt = f"""You are simulating how a specific person would rate a comment's
offensiveness. This person is a {persona['age']}-year-old {persona['gender']}
from {persona['country']}, whose moral values are: {persona['moral_values']}.
Rate the comment as this person would, from 0 (not offensive) to 4 (extremely offensive)."""
```

A few things worth knowing before you build this out, straight from your own literature review:

- **Kamruzzaman et al.'s finding is directly actionable here**: alignment with real humans peaks at 1–3 demographic attributes in the prompt and *degrades* with more. Don't assume "give the model everything I know about the annotator" is better — test attribute count as a variable, don't just maximize it.
- **CreHate's finding**: telling the model "consider this from a Singaporean perspective" as an instruction, without actually changing the underlying architecture, mostly doesn't move predictions. Don't expect persona prompting alone to fully solve cultural adaptation — it's a useful signal, but budget time to also try structural approaches (section 7).
- **Run each persona multiple times if you want variance, not just a point estimate.** A single call at temperature 0 gives you the model's mode; if you want to simulate the actual *spread* a demographic group would produce, you need either repeated sampling at temperature > 0, or the logprob approach below.

---

## 5. Getting distributional (soft) predictions — this is the technical crux of the disagreement-modeling papers

You have three options here, in increasing order of sophistication:

**Option 1 — Repeated sampling.** Call the same prompt N times at temperature > 0 (e.g., 0.7–1.0), collect the distribution of labels returned, treat the empirical frequency as your predicted distribution. Simple, but expensive (N× the API cost) and noisy at low N.

**Option 2 — Logprobs (cheaper and more principled).** If your label space is small (e.g., a rating from 0–4, or a yes/no), you can request the token-level log-probabilities the model assigns to each possible answer token in a *single* API call, and convert them directly into a probability distribution — no repeated sampling needed.

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    max_tokens=1,
    logprobs=True,
    top_logprobs=5,   # returns the top 5 candidate tokens with their log-probs
)
top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
# convert to a probability distribution over your label set
import math
probs = {t.token: math.exp(t.logprob) for t in top_logprobs}
```

This is the technique underlying a lot of the "does the model show doubt the way humans do" analyses in the overconfident-judge paper — comparing the *entropy* of this distribution to human label entropy is a very direct way to test their finding on your own data.

**Option 3 — Ensemble of personas.** Run the persona-prompting method (section 4) across many distinct simulated annotators (ideally matching the actual demographic/moral composition of your D3CODE sample), and treat the resulting label distribution as your model's prediction. This is what Movva et al. and Sarumi et al. do, and it's the most direct simulation of "would these two groups disagree."

---

## 6. Evaluating against real human disagreement

Once you have a predicted distribution (from any of the above), you need metrics that compare *distributions*, not just accuracy on a single label. These are the standard ones in this literature (and used in the LeWiDi shared task, which is worth reading the scoring scripts of even if you don't use their datasets):

| Metric | What it tells you | Notes |
|---|---|---|
| **Cross-Entropy (CE) / soft-label loss** | How well your predicted distribution matches the human label distribution | Primary LeWiDi metric — lower is better |
| **Jensen-Shannon Divergence (JSD)** | Symmetric distance between two distributions | Good complement to CE, bounded [0,1] |
| **Krippendorff's α** | Inter-annotator agreement, computed on human data *and* on your simulated annotator pool | Comparing real α to simulated α is exactly how Sarumi et al. showed LLMs produce artificially uniform "consensus" |
| **Entropy comparison** | Whether your model shows appropriate uncertainty on ambiguous items | This is your direct test of the "overconfident judge" finding |
| **MAE / correlation vs. continuous score** | If you're predicting a continuous quantity (e.g., D3CODE's raw 0–4 rating averaged per group) | Simple and interpretable |
| **Group-disagreement correlation** | Correlation between predicted and actual *pairwise disagreement rates* between specific demographic/cultural groups | This is Movva et al.'s key metric, and probably your single most important one — it's the most direct measurement of "predicting who disagrees with whom" |

Practical tip: implement these once as a small evaluation library early on (`krippendorff` package exists on PyPI; `scipy.stats.entropy` gives you both Shannon entropy and, with two distributions, KL divergence for JSD) — you'll reuse it across every model/method you test.

---

## 7. Beyond prompting: fine-tuned architectures (if you want to go past "LLM as annotator")

Prompting is fast to iterate on but is fundamentally limited by whatever the base model already "knows." Several papers go further and actually train a model on the annotation data itself:

- **Jury Learning (Gordon et al.)**: train a model to predict *individual* annotator labels (not aggregate), typically by adding annotator ID or annotator demographic features as additional input to a classifier (e.g., concatenated to the text embedding, or via a separate embedding table per annotator). At inference time, you can then simulate a custom "jury" by sampling/weighting which simulated annotators you aggregate over. This is a fine-tuning task, not just prompting — a reasonable starting architecture is a BERT/RoBERTa-based classifier with the text encoded normally and demographic/annotator-ID features concatenated before the final classification head.
- **Demographic-aware Mixture-of-Experts (DeM-MoE style)**: instead of one shared classifier head, route each prediction through a small set of "expert" sub-networks, each specializing in a demographic subgroup, gated by the annotator's demographic profile. More complex to implement but directly targets the "different groups have qualitatively different decision patterns" finding from Maurer and GRASP.
- **Multi-task learning with a moral-values auxiliary task**: since D3CODE gives you MFQ-2 scores per annotator, you could train a model to jointly predict (a) the offensiveness rating and (b) the annotator's moral foundation scores, sharing a representation — this directly operationalizes the "moral values mediate cultural disagreement" finding rather than just prompting around it.

These are standard supervised fine-tuning setups (`transformers` + `Trainer`, or a custom PyTorch training loop) — nothing exotic architecturally, the novelty is entirely in *what features you condition on* and *what you evaluate against*. This can run on a single consumer/Colab GPU for a BERT-sized model; you don't need anything exotic compute-wise unless you're fine-tuning something LLM-sized (in which case look at parameter-efficient methods like LoRA rather than full fine-tuning).

---

## 8. Ensembling (a good default final step)

Atil et al.'s finding — no single prompting/persona strategy wins everywhere, but a simple learned ensemble beats all of them — is easy to replicate and worth doing regardless of what else you build:

1. Generate predictions from several different methods (zero-shot, few-shot, 2–3 different persona strategies, maybe 2 different base models).
2. Treat each method's output as a feature.
3. Train a simple model (logistic regression or SVM, nothing fancy) on top of these features to predict the true label/distribution.

This is cheap to build once you have the individual predictions and gives you a strong, easily-defensible final number to report.

---

## 9. The interpretability angle (optional, more advanced)

Two papers in your notes go further into *why* a model gives the rating it does — worth knowing about even if you don't implement them:

- **Token Distribution Dynamics (Feng et al.)**: identifies which specific tokens in the input are driving the model's offensiveness judgment, by tracking how token-level output distributions shift as you mask/perturb parts of the input. Reproducible with open-weight models where you have access to internal activations (needs `transformers` + a local/hosted open model, not just an API — closed APIs like GPT-4 don't expose this).
- **Activation steering (Ghandeharioun et al.)**: directly manipulating an internal "perspective" representation to shift the model's judgment. This needs full white-box access (open-weight models like Llama via `transformers` + hooks, or a library like `TransformerLens`), so it's not compatible with GPT-4/Claude via API — only relevant if you decide to work with an open model for part of the thesis.

If your timeline is tight, these are reasonable to cite as "future work directions" rather than implement — they require meaningfully more engineering (white-box model access) than the prompting/fine-tuning pipeline above.

---

## 10. A concrete end-to-end pipeline for D3CODE

Putting it together, a realistic build order:

1. **Load and split D3CODE** (`items.csv`, `raters.csv`, `ratings.csv`) — decide on your train/dev/test split, being careful to check whether you should split by item, by annotator, or both (splitting only by item, with the same annotators appearing in train and test, would leak annotator-specific patterns and inflate performance).
2. **Baseline**: majority-vote / mean-rating prediction with no LLM at all. You need this number to prove any LLM-based method adds value.
3. **Paradigm A baseline**: zero-shot and few-shot GPT-4/Claude prediction against the mean human rating. Report standard accuracy/MAE.
4. **Paradigm B, persona ensembling**: for a sample of items, generate persona-conditioned predictions matching D3CODE's actual annotator distribution (sample personas proportionally from `raters.csv`, including country/region and MFQ profile), aggregate into a predicted distribution, evaluate with CE/JSD/entropy against the real distribution.
5. **Fine-tuned model** (if time allows): train a classifier with annotator/demographic features as in section 7, compare against the prompting-only approaches.
6. **Cross-cultural correlation check**: compute predicted vs. actual pairwise disagreement rates between specific country/region pairs (Movva-style) — this is your most thesis-relevant single result.
7. **Ensemble** (section 8) as your final reported number, alongside the individual methods so the improvement is visible.

---

## 11. Tooling checklist

- `openai` and/or `anthropic` Python SDKs — API-based prompting, no GPU needed
- `transformers` + `datasets` (HuggingFace) — only needed if you fine-tune or need white-box access
- `torch` — only needed for fine-tuning
- `scikit-learn` — for the ensembling step and simple baselines
- `krippendorff` (PyPI package) — agreement statistics
- `scipy.stats` — entropy, KL divergence
- A **caching layer** for API calls (even a simple local SQLite or JSON cache keyed by prompt hash) — you will re-run experiments, and re-paying for identical calls is wasteful and slows iteration
- A **spend limit / budget alert** set on your API account before running anything at full scale

---

## 12. Common pitfalls worth knowing in advance

- **Non-determinism**: even at `temperature=0`, API-based LLMs aren't perfectly deterministic run-to-run (this is a known property of how these models are served). Don't be surprised by small variation between identical calls — report it rather than assuming a bug.
- **Prompt sensitivity**: small wording changes can shift results meaningfully. Keep a fixed prompt template per experiment condition and change only the variable you're testing (this is literally the point of several of your papers, so treat your own prompts with the same scrutiny).
- **Data contamination**: D3CODE draws its items from the Jigsaw dataset, which is old and public — there's a real possibility GPT-4-class models have seen this exact data during training. Worth at least mentioning as a limitation, and possibly testing on the CP (cultural posts) subset separately since that's less likely to be memorized than the SBIC-derived portion.
- **Refusals silently skewing your data**: if a model refuses or gives a non-answer on some fraction of offensive items, and you drop those rows, you're systematically dropping the most offensive content — track refusal rate as its own metric, don't just discard.
