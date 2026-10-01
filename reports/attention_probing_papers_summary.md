# Paper Summaries & Attention Probing Findings

For your supervisor: paper breakdown structured around exactly what she asked (goals, probing/steering methods, libraries, LLMs, technical details), followed by answers to her two initial questions using our own attention-probing results so far.

---

## Part 1 — The Six Papers

| # | Paper | Goal | LLMs Used | Libraries / Tools | Probing Method | Steering Method | Key Technical Detail |
|---|---|---|---|---|---|---|---|
| 1 | Multilingual Political Views: Identification & Steering | Show political stance is linearly decodable across languages; steer it cross-lingually | LLaMA-3.1 (8B/70B), Qwen-3 (8B/14B/32B), Aya-Expanse | PyTorch, HF `transformers`/`datasets`, Political Compass Test | Logistic regression probes on **every attention head's value output**; peak 90% acc. at Layer 30/Head 19 (also 11, 13, 26–31 in LLaMA-3.1-8B) | DiffMean vector on top-K heads, injected via forward hook on head value outputs: `value += α·σ·v` | Hooks on individual head outputs, not the whole residual stream |
| 2 | FairSteer: Inference-Time Debiasing | Mitigate social bias at inference, no fine-tuning | Llama-2-chat, Llama-3-instruct (8B), Vicuna-1.5, Mistral-v0.3 | PyTorch, HF `evaluate`, BBQ, MMLU, CrowS-Pairs | Linear classifier on **last-token residual stream activation** per layer; Layers 13–15 peak (>90% acc.) | Only steers **when a classifier flags bias** (dynamic trigger): `a_adj = a + v` at chosen layer | Conditional/triggered steering, not always-on |
| 3 | Steering Towards Fairness (SVE) | Neutralize political bias across English/Urdu/Punjabi | Mistral-7B-Instruct, DeepSeek-VL-7B-Chat | PyTorch, HF `transformers`, multilingual PCT | Logistic regression on last-token hidden states per layer | **Ensemble** of steering vectors across layers {8,12,16,20,24}, quality-weighted and combined before injection | Multi-layer > single-layer for stability/fluency |
| 4 | Probing Political Ideology: Generalization | Test if ideological probe directions causally generalize to bias/voting/neutralization tasks | LLaMA-2-7B-chat, LLaMA-3.1-8B-instruct, Qwen-2.5-7B-instruct | PyTorch, HF `transformers`, scikit-learn (Ridge) | Ridge regression per attention head, predicting a continuous ideology score | Steer top-k heads (ranked by probe R²): `x += α·σ·θ`, α ∈ [-30,30] | Same architecture family/models as our project — directly comparable |
| 5 | **PASTA: Post-Hoc Attention Steering** | Force the model to attend more to user-specified tokens, no retraining | LLaMA-7B/13B, GPT-J-6B, Vicuna-7B | PyTorch, HF `transformers` | Profiles which heads' *raw attention scores* most affect task accuracy on small (200–1000 sample) sets; keeps top ~50–100 heads | **Directly reweights the softmax attention matrix**: down-weight non-target tokens by α≈0.01, renormalize | Operates on raw `softmax(QKᵀ/√d)`, not on activations/values — the only paper doing this |
| 6 | TDD: Token Distribution Dynamics | Project internal activations into vocabulary space to find "toxic trigger" tokens | GPT-2, GPT-J-6B, BLOOM-7B, Pythia-6.9B, LLaMA2-7B/13B, OPT-30B | PyTorch, HF `transformers`, RealToxicityPrompts, Perspective API | Pass any token's hidden state through `model.lm_head` to read its vocabulary-space "opinion" at that layer | Identify top-15%-saliency toxic tokens, replace with space tokens | Cheap diagnostic — needs no probe training at all |

### What's common across all six (transferable to our setup)
- **Two families of method**: (a) papers 1–4 probe/steer *activations* (residual stream or attention-head value outputs) via a DiffMean/linear-probe direction; (b) paper 5 (PASTA) probes/steers *raw attention scores* directly; (c) paper 6 sidesteps probing entirely via the LM head.
- **Mid-to-late layers dominate**: every activation-probing paper finds peak accuracy in roughly layers 11–31 of an ~32-layer model (paper 4 uses the exact same LLaMA-3.1-8B-instruct we do — directly comparable layer indices).
- **Small contrastive sets are enough**: 100–1000 contrastive pairs suffice to build a usable steering vector in every paper — consistent with your supervisor's "no training necessary."
- **All hook points map onto reusable code**: residual stream → `model.model.layers[l]`; head value outputs → `self_attn.o_proj`; raw attention scores → `output_attentions=True` (what we're already using).

---

## Part 2 — Answering Your Supervisor's Two Questions

*(Using our own `attention_probe_detailed.py` results: Llama-3.1-8B-Instruct, 70 items stratified across D3CODE's 3 categories, raw `output_attentions`, sink-excluded/renormalized to allow fair cross-condition comparison — see conversation for why the raw numbers need this correction.)*

### Q1: "Where does the LLM pay attention in a simple prompt?"

- **~80% of all attention** (at the final token, mean over layers/heads) goes to a handful of fixed chat-template tokens (`<|begin_of_text|>`, header tokens) — a well-documented **attention sink** effect (Xiao et al., 2023), largely independent of prompt content, and present at every layer including layer 0. Confirmed both numerically and by inspecting bertviz heatmaps directly.
- Of the **remaining ~20% "content" attention** on a plain baseline prompt (no demographic info): roughly **87% goes to the fixed task instruction** ("Is this text offensive? Answer with only 0 (no) or 1 (yes).") and only **~13% goes to the item text itself**.
- **Takeaway:** in a bare classification prompt, the decision-relevant attention is dominated by the task-framing instruction, not by the text being judged.

### Q2: "How does attention change when we add demographic information?"

- The demographic phrase itself captures **5.9%–9.9% of content attention**, depending on which descriptor is used:

  | Descriptor | Share of content attention |
  |---|---|
  | "You are a man." | 5.9% |
  | "You are a woman." | 6.3% |
  | "You are Christian." | 5.9% |
  | "You are Muslim." | 7.5% |
  | "...Western Europe." | 7.1% |
  | "...Arab Culture region." | 9.9% |

- This new attention comes **almost entirely out of the instruction's share** (87.0% → 76.8–81.6% across conditions), while the item-text share stays essentially flat (12.4–13.4%). So adding demographic info does **not** distract the model from the text it's judging — it competes with the task-framing instruction instead.
- The effect is **not symmetric**: "Arab Culture" draws ~40% more relative attention than "Western Europe" (9.9% vs 7.1%); "Muslim" draws ~27% more than "Christian" (7.5% vs 5.9%); gender is essentially balanced (5.9% vs 6.3%).
- Raw bertviz heatmaps cannot visibly show this — the sink is too large relative to the effect — so we built a sink-excluded, renormalized bar chart per example item as the visual counterpart of the table above.
- Your third point ("try changing the prompt and see whether attention scores change") is what the 7-condition design already does — each item is run under all 7 phrasings and compared item-by-item (paired design), not just as separate averages.

---

## Part 3 — How to Proceed

**This phase (probing) is essentially answered** — the deliverables above (segment table, per-item paired deltas, sink-excluded bar charts) directly cover both of your supervisor's questions and are ready to present.

**For the steering phase** (not started yet, per her "steering excluded for now" scoping): her own phrasing — *"manipulating the attention toward the demographic information"* — points specifically at **PASTA (paper 5)**, not the DiffMean-style papers (1–4). The distinction matters:
- Papers 1–4 steer by adding a vector to **activations** (residual stream or head *value* outputs) — this changes what the model "thinks," not literally where it looks.
- PASTA reweights the **raw attention score matrix** itself — literally turning up attention on chosen tokens (e.g., the demographic phrase) — which is a much closer match to what she described, and it operates on exactly the same `output_attentions`-style data our probing step already extracts.

Concrete next steps once she gives the go-ahead to start steering:
1. Read the PASTA repo (`github.com/QingruZhang/PASTA`) closely to confirm how it integrates with Hugging Face's attention computation for LLaMA — this needs verifying before writing code, since PASTA's original models predate the `eager`/SDPA attention-implementation split we already had to work around for plain probing.
2. Identify which heads to target: either reuse PASTA's own profiling method (small held-out set, pick heads whose reweighting most changes accuracy), or start simpler — target the heads that already show the largest demographic-attention shift in our current per-token data.
3. Apply PASTA's reweighting formula to the demographic phrase span (which we already locate via `find_token_span` for every item/condition) and measure whether the predicted label (0/1) flips — directly testing her causal question, "does changing attention change the prediction?"
4. Keep the DiffMean/activation-steering approach (papers 1, 3, 4) as a documented fallback if PASTA's attention-level reweighting proves difficult to wire into Llama-3.1's current `transformers` implementation.
