# Paper Summaries: Literature Cited in `literature_comparison_report.md`

All content below comes from the papers' own abstracts/full text (fetched directly, not just search snippets), cross-checked against what I actually used each paper for in the comparison report.

---

* **In-Context Impersonation Reveals Large Language Models' Strengths and Biases** ✔️
Year: 2023
Author(s): Leonard Salewski, Stephan Alaniz, Isabel Rio-Torto, Eric Schulz, Zeynep Akata (NeurIPS 2023, Spotlight)
Main Message:
LLMs can adopt personas in-context, and doing so measurably changes their behavior on tasks unrelated to the persona itself, revealing both genuine capability gains (expertise personas) and stereotype-driven biases (identity personas), from the same underlying mechanism.
Experiment:
Three tasks with persona-prefixed prompts: (1) a multi-armed bandit task with LLMs impersonating children at different developmental ages, (2) language-based reasoning with domain-expert vs. non-expert personas, (3) a visual description task with LLMs impersonating experts in different domains (e.g. "bird expert," "car expert") describing images.
Key Findings:
- **Persona-appropriate skill transfer:** models impersonating children of different ages reproduce human-like developmental exploration patterns; domain-expert personas outperform non-expert personas on matching reasoning tasks.
- **Beneficial impersonation:** a "bird expert" persona describes birds better than a "car expert" persona describes the same birds.
- **Stereotype-driven bias:** a "man" persona describes cars better than a "woman" persona, despite car-description having nothing to do with gender, an identity statement alone shifted task performance in a stereotype-consistent direction.
*Used for comparison:* Finding 1 (demographic framing reliably shifts model behavior on an unrelated task) — this is the closest direct precedent: a one-sentence identity statement, nothing else changed, produces a systematic, content-relevant behavioral shift, exactly the mechanism behind our own placebo-controlled result.

---

* **Bias Runs Deep: Implicit Reasoning Biases in Persona-Assigned LLMs** ✔️
Year: 2024
Author(s): Shashank Gupta, Vaishnavi Shrivastava, Ameet Deshpande, Ashwin Kalyan, Peter Clark, Ashish Sabharwal, Tushar Khot (ICLR 2024)
Main Message:
LLMs harbor deep-rooted bias against socio-demographic groups underneath a veneer of fairness: they overtly reject stereotypes when asked directly, but manifest stereotypical and erroneous reasoning when simply asked to answer while adopting a persona.
Experiment:
24 reasoning datasets x 4 LLMs (including ChatGPT-3.5, GPT-4-Turbo) x 19 personas spanning 5 socio-demographic groups (e.g. "an Asian person," "a physically disabled person"). Each dataset run with and without a persona prefix, comparing accuracy and error patterns.
Key Findings:
- **Ubiquitous:** 80% of personas demonstrate bias in ChatGPT-3.5; GPT-4-Turbo is less biased but still shows it in 42% of personas.
- **Significant:** some datasets show 70%+ performance drops under a persona; some personas suffer statistically significant drops on 80%+ of datasets.
- **Two failure modes:** models either abstain from answering (58% of errors for the physically-disabled persona in ChatGPT-3.5 are abstentions) or make silent reasoning errors without stating a stereotype outright.
- **Hard to fix:** de-biasing prompts have minimal to no effect on reducing persona-induced errors.
*Used for comparison:* Finding 1 (same mechanism, "veneer of fairness" framing fits our result: nothing in our prompt mentions offensiveness policy, the identity clause alone does the work) and Finding 3 (their framing that bias appears "underneath" stated fairness is the closest conceptual cousin to our placebo pushing the opposite direction, i.e. the model's behavior decouples from a neutral default under *any* persona framing, not just identity ones).

---

* **Calibrate Before Use: Improving Few-Shot Performance of Language Models** ✔️
Year: 2021
Author(s): Tony Z. Zhao, Eric Wallace, Shi Feng, Dan Klein, Sameer Singh (ICML 2021)
Main Message:
GPT-3/GPT-2's few-shot accuracy is unstable across prompt format, example choice, and example order, not because the model lacks the knowledge, but because of systematic, content-independent biases baked into how it reads a prompt. A cheap, data-free calibration step neutralizes most of that instability.
Experiment:
Identify three biases (majority label bias, recency bias, common token bias) across standard few-shot classification tasks, then correct for them via "contextual calibration": feed a content-free input (literally the string "N/A") through the same prompt, measure the resulting (biased) label distribution, and use it to rescale the model's real predictions back toward uniform.
Key Findings:
- **Majority label bias:** the model favors whichever answer appeared most often in the provided examples.
- **Recency bias:** the model favors whichever answer appeared nearest the end of the prompt, independent of content.
- **Common token bias:** predictions skew toward tokens common in pre-training data regardless of task relevance.
- **Contextual calibration** (the content-free-input technique) improves average accuracy by up to 30.0 percentage points and substantially reduces prompt-choice variance across GPT-3 and GPT-2.
*Used for comparison:* Finding 1/3 (our placebo condition is structurally the same move as their content-free "N/A" input, a control for what a prompt does when it has no relevant content, just applied to a bias question instead of a calibration question) and Finding 5 (their named "recency bias" is the exact mechanism we invoked to explain why our `suffix` prompt position collapsed all content-specific signal into a uniform shift).

---

* **Persistent Anti-Muslim Bias in Large Language Models** ✔️
Year: 2021
Author(s): Abubakar Abid, Maheen Farooqi, James Zou (AAAI/ACM AIES 2021)
Main Message:
GPT-3 captures a persistent, unusually severe Muslim-violence association that appears consistently across very different elicitation methods, and is not easily removed by ordinary positive-framing mitigation.
Experiment:
Three separate methods for probing the same association in GPT-3: (1) prompt completion (does the model continue a Muslim-related prompt with violent content), (2) analogical reasoning (what word does the model map "Muslim" to, analogous to how it maps other religions), (3) open-ended story generation (does violent content appear when a Muslim character is introduced).
Key Findings:
- **Severity relative to other groups:** "Muslim" is analogized to "terrorist" in 23% of test cases, versus "Jewish" to its most common stereotype ("money") in only 5% of cases, a roughly 4-5x larger effect for this one category.
- **Consistency across methods:** the same bias shows up whether probed via completion, analogy, or story generation, it isn't an artifact of one elicitation technique.
- **Mitigation is partial:** adding the six most positive adjectives to the context reduces violent completions for "Muslim" from 66% to 20%, a real reduction, but still elevated relative to baseline rates for other religions.
*Used for comparison:* Finding 2 (Arab-Culture/Muslim asymmetry) — this is the closest direct precedent for our result: one specific identity category drawing a disproportionate negative association relative to comparably-tested alternatives, replicated across methods, and only partially removable by simple mitigation. Ours replicates the same qualitative asymmetry on a newer model (Llama-3.1-8B-Instruct, 2026) via a completely different method (attention share + calibrated probability shift rather than completion sampling).

---

* **BBQ: A Hand-Built Bias Benchmark for Question Answering** ✔️
Year: 2022
Author(s): Alicia Parrish, Angelica Chen, Nikita Nangia, Vishakh Padmakumar, Jason Phang, Jana Thompson, Phu Mon Htut, Samuel R. Bowman (ACL 2022 Findings)
Main Message:
Question-answering models rely on social stereotypes to fill in gaps when context is ambiguous, and that reliance doesn't fully disappear even when the context becomes informative enough to answer correctly without stereotyping.
Experiment:
58,492 hand-built multiple-choice questions across 9 social dimensions (age, disability, gender, nationality, physical appearance, race/ethnicity, religion, socio-economic status, sexual orientation, plus 2 intersectional axes), each presented in both an ambiguous version (not enough context to answer correctly without guessing) and a disambiguated version (enough context to answer correctly).
Key Findings:
- Under ambiguous context, models consistently reproduce the stereotype-aligned answer rather than saying "unknown."
- Under disambiguated context, accuracy improves overall, but models still show up to 3.4 percentage points higher accuracy when the correct answer happens to align with a social bias than when it conflicts with one, widening to over 5 points for some gender-related examples.
*Used for comparison:* Finding 2 (background support) — BBQ's religion and race/ethnicity axes are cited as corroborating evidence that Muslim- and Middle-Eastern-coded categories show up as disproportionately stereotyped across many models and benchmarks, not just in one paper's specific setup.

---

* **Learning from Disagreement: A Survey** ✔️
Year: 2021
Author(s): Alexandra N. Uma, Tommaso Fornaciari, Dirk Hovy, Silviu Paun, Barbara Plank, Massimo Poesio (Journal of Artificial Intelligence Research, Vol. 72)
Main Message:
The standard NLP/CV assumption that a single "gold" label exists for every item is often wrong; genuine, persistent human disagreement is common even on tasks assumed to be objective, and the field needs learning methods that use that disagreement as signal instead of discarding it via majority-vote aggregation.
Experiment:
Not a single experiment, a survey. It reviews the evidence for disagreement across NLP and computer vision tasks (from POS tagging to image classification to natural language inference) and systematically compares three families of approach for training on disagreement-containing data: aggregation into consensus labels, soft labels (keeping the full distribution over annotator judgments), and multi-annotator models (explicitly modeling individual annotators or annotator groups).
Key Findings:
- Which method wins depends on data quality: **training directly on soft labels beats training on aggregated/majority-vote labels when a dataset has many high-quality annotations per item**; when annotation quality is limited, blending gold and soft labels performs best under harder evaluation conditions.
- The field's evaluation methodology itself shapes which approach looks best, so results aren't method-independent.
*Used for comparison:* Finding 4 (methodological) — establishes the "disagreement-as-signal, not noise" framework our item-selection method (entropy over the full rating distribution) sits inside; used to show our baseline-ceiling confound is a genuine gap, since this survey's methods for combining disagreement-aware selection with a *model's own behavior* under that selection aren't addressed here.

---

* **The "Problem" of Human Label Variation: On Ground Truth in Data, Modeling and Evaluation** ✔️
Year: 2022
Author(s): Barbara Plank (EMNLP 2022)
Main Message:
Human label variation should not be treated as noise to be eliminated; it stems from legitimate sources (genuine annotator disagreement, subjective interpretation, multiple valid answers, task ambiguity) and affects every stage of the ML pipeline (data, modeling, evaluation) simultaneously, not just one stage in isolation.
Experiment:
A position/synthesis paper, not a single experiment. It reconciles competing prior definitions of "label variation," surveys datasets that publish un-aggregated (rather than majority-voted) labels, and maps out where current approaches fall short across the full pipeline.
Key Findings:
- Proposes treating variation-handling as a **joint** problem across data collection, modeling, and evaluation, rather than three separate fixes.
- Calls for wider release of un-aggregated, per-annotator labels as a research resource (most published datasets discard this at collection time).
- Argues current evaluation practice (comparing to one majority-vote "ground truth") is itself part of the problem, since it can only ever penalize a model for correctly reflecting real human disagreement.
*Used for comparison:* Finding 4 (methodological), alongside Uma et al., as the second half of the "disagreement modeling" literature our item-selection method draws on, and as the paper making explicit the point our own confound illustrates concretely: evaluation choices (here, entropy-based item selection feeding into probability-shift measurement) can silently interact with modeling choices (the model's own baseline confidence) in ways that aren't visible until you check.

---

* **Lost in the Middle: How Language Models Use Long Contexts** ✔️
Year: 2023 (arXiv), TACL 2024
Author(s): Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang
Main Message:
Language models don't use long input contexts uniformly: they access information best when it's at the very beginning or very end of the context, and struggle when it's buried in the middle, and this holds even for models explicitly built for long contexts.
Experiment:
Multi-document question answering and key-value retrieval tasks, where the position of the one relevant document/pair within the input is systematically varied (tested at positions 1, 5, 10, 15, 20 of 20 total), across GPT-3.5, GPT-4, Claude, and several open models.
Key Findings:
- **U-shaped performance curve:** accuracy is highest when the relevant document is first, nearly as high when it's last, and drops sharply (20-30+ percentage points) when it's in the middle.
- This holds regardless of how "long-context" the model claims to be, it's not fixed by simply supporting a longer context window.
*Used for comparison:* Finding 5 (prompt position matters, non-monotonically) — a structural cousin of our result: theirs is a U-shape across a long document list, ours is a sharp break between "earlier in a short prompt" and "the very last thing before the model must answer," but both establish the same underlying point, position within a prompt is not a neutral variable, and its effect isn't simply "closer to the salient content = bigger effect."

---

## Sources

- [In-Context Impersonation Reveals Large Language Models' Strengths and Biases](https://arxiv.org/abs/2305.14930)
- [Bias Runs Deep: Implicit Reasoning Biases in Persona-Assigned LLMs](https://arxiv.org/abs/2311.04892)
- [Calibrate Before Use: Improving Few-Shot Performance of Language Models](https://arxiv.org/abs/2102.09690)
- [Persistent Anti-Muslim Bias in Large Language Models](https://arxiv.org/abs/2101.05783)
- [BBQ: A Hand-Built Bias Benchmark for Question Answering](https://arxiv.org/abs/2110.08193)
- [Learning from Disagreement: A Survey](https://www.jair.org/index.php/jair/article/view/12752)
- [The "Problem" of Human Label Variation: On Ground Truth in Data, Modeling and Evaluation](https://arxiv.org/abs/2211.02570)
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
