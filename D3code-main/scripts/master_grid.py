"""
Master grid, v2 design: every condition under every prompt variant, on one
frozen item list shared by all models.

What changed from the LRZ (v1) grid, and why
  - Items: read from the frozen list written by select_items.py (whole
    D3CODE pool, in band for BOTH models, same N, same 50/50 selection/test
    split). v1 filtered per model and ended with 236 vs 64 items.
  - Full factorial instead of blocks: all 22 conditions x all 10 variants, so
    every comparison has the same items and the same set of conditions.
  - Phrases are built from one noun per condition through fixed templates
    ("You are {noun}.", "Imagine you are {noun}.", "As {noun}, ..."), so a
    demographic and a placebo differ only in the noun.
  - Placebos: 8 content-free nouns, two per token length the demographics
    need (2, 4, 5, 6 tokens). Every demographic gets its exact-length placebos
    as matched partners. Llama and Qwen tokenize all of these to the same
    length in every variant (checked by check_matching.py), so the matching is
    identical across models too.
  - Context variants use ONE elaboration sentence for demographics and
    placebos alike (v1 had "offensive" vs "refreshing"). The demographic span
    is the identity sentence only; the shared elaboration is not part of it.
  - Log-odds of "1" vs "0" are recorded next to P(yes): no saturation at 0/1.
  - Value-weighted attention (Kobayashi et al. 2020) per head next to raw
    attention: attention x ||W_O^h v_j||, renormalized per head.
  - Noise check at the end: the first NOISE_ITEMS items are run again and
    compared to their grid rows, and grid baselines are compared to the
    select_items.py scores (different code path: no output_attentions).

Variants (demographic span in brackets):
  prefix        [You are X.] Text: "..." Is this text offensive? ...
  reworded      [Assume you are X.] Text: ...          (paraphrase templates,
  imagine       [Imagine you are X.] Text: ...          all in prefix position)
  consider      [Consider that you are X.] Text: ...
  infix         Text: "..." [You are X.] Is this text offensive? ...
  suffix        Text: "..." Is this text offensive? ... [You are X.]
  embedded      Text: "..." As [X], is this text offensive? ...
  target        Text: "..." Do you think this text is offensive for [X]? ...
  context       [You are X.] <elaboration> Text: ...
  context_long  [You are X.] <elaboration> Answer from that perspective. Text: ...

Per-layer and per-head attention is stored for every row (npz), at no extra
forward passes. A tokenizer-only preflight aborts before the model loads if
any span is unlocatable or any matched pair is not exact.
"""
import os
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use("Agg")   # imported at module load so a missing install fails at start, not mid-run
import matplotlib.pyplot as plt

# torch >= 2.13 routes some eager CUDA ops through experimental Triton / CuTe-DSL
# kernels that need gcc at runtime (this killed the LRZ smoke test). Switching the
# overrides off sends those ops back to the regular ATen path. No-op elsewhere.
for _mod in ("triton_utils", "cutedsl_utils"):
    try:
        getattr(__import__("torch._native", fromlist=[_mod]), _mod).deregister_op_overrides()
        print(f"torch native {_mod} op overrides disabled", flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f"(torch native {_mod} override disable skipped: {_e!r})", flush=True)

from span_utils import find_token_span, find_span_by_delimiters
from zeroshot_two_datasets import PROMPT

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
MODEL_TAG = os.environ.get("MODEL_TAG", MODEL_NAME.rstrip("/").split("/")[-1].replace(".", "-"))
MAX_LENGTH = 1024
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results/v2/master_grid")
ITEMS_CSV = os.environ.get("ITEMS_CSV", "items/items_v2.csv")
SCORES_DIR = os.environ.get("SCORES_DIR", "results/v2/scores")
MAX_ITEMS = int(os.environ.get("MAX_ITEMS", "0"))        # >0: smoke test on the first N items
if MAX_ITEMS:
    RESULTS_DIR = RESULTS_DIR + "_smoke"
PREFLIGHT_ONLY = os.environ.get("PREFLIGHT_ONLY", "0") == "1"
NOISE_ITEMS = int(os.environ.get("NOISE_ITEMS", "20"))
FIGURES_DIR = os.environ.get("FIGURES_DIR", "figures/v2")
N_ILLUSTRATIVE = 3
CHECKPOINT_EVERY = 5000

SEGMENTS = ["demographic", "instruction", "item_text", "other"]
SEGMENT_COLORS = {"demographic": "tab:red", "instruction": "tab:blue", "item_text": "tab:green"}

INSTRUCTION_TEXT = "Is this text offensive? Answer with only 0 (no) or 1 (yes)."
INSTRUCTION_TEXT_EMBEDDED = "is this text offensive? Answer with only 0 (no) or 1 (yes)."
TARGET_BEFORE = "Do you think this text is offensive for "
TARGET_AFTER = "? Answer with only 0 (no) or 1 (yes)."
TEXT_PREFIX, _ = PROMPT.split("{text}")
TEXT_SUFFIX = '"\n'
# One elaboration for demographics and placebos alike, so the context variants
# differ between conditions only in the identity sentence.
ELABORATION = "Your background and everyday experience shape how you read things."
PERSPECTIVE = "Answer from that perspective."

# --- conditions ---------------------------------------------------------------
# name -> (noun, axis, is_placebo). Every phrase is built from the noun through
# the templates below, identically for demographics and placebos.
CONDITIONS = {
    # all 8 D3CODE Region values, verbatim
    "region_western_europe": ("someone from Western Europe", "region", False),
    "region_north_america": ("someone from North America", "region", False),
    "region_latin_america": ("someone from Latin America", "region", False),
    "region_arab_culture": ("someone from the Arab Culture region", "region", False),
    "region_indian_cultural_sphere": ("someone from the Indian Cultural Sphere", "region", False),
    "region_sinosphere": ("someone from the Sinosphere", "region", False),
    "region_sub_saharan_africa": ("someone from Sub Saharan Africa", "region", False),
    "region_oceania": ("someone from Oceania", "region", False),
    "gender_man": ("a man", "gender", False),
    "gender_woman": ("a woman", "gender", False),
    "religion_christian": ("a Christian", "religion", False),
    "religion_muslim": ("a Muslim", "religion", False),
    "identity_arab": ("an Arab", "identity", False),
    "identity_european": ("a European", "identity", False),
    # content-free placebos, two per demographic noun length (2, 4, 5, 6 tokens
    # under both tokenizers). Hobbies and habits only: no group, origin, belief,
    # occupation or trait a D3CODE rater demographic could encode.
    "placebo_runner": ("a runner", "placebo", True),
    "placebo_cyclist": ("a cyclist", "placebo", True),
    "placebo_early_riser": ("an early riser", "placebo", True),
    "placebo_chess": ("someone who plays chess", "placebo", True),
    "placebo_maps": ("someone who collects old maps", "placebo", True),
    "placebo_crossword": ("someone who enjoys crossword puzzles", "placebo", True),
    "placebo_stairs": ("someone who usually takes the stairs", "placebo", True),
    # the placebo every v1 study used; kept for continuity. It was the outlier
    # placebo in v1 and "tea over coffee" may read as a cultural cue, so check it
    # separately before pooling it with the others.
    "placebo_tea": ("someone who prefers tea over coffee", "placebo", True),
}
DEMOGRAPHICS = [c for c, v in CONDITIONS.items() if not v[2]]
PLACEBOS = [c for c, v in CONDITIONS.items() if v[2]]

SENTENCE_TEMPLATES = {          # identity sentence per variant
    "prefix": "You are {n}.", "reworded": "Assume you are {n}.",
    "imagine": "Imagine you are {n}.", "consider": "Consider that you are {n}.",
    "infix": "You are {n}.", "suffix": "You are {n}.",
    "context": "You are {n}.", "context_long": "You are {n}.",
}
VARIANTS = ["prefix", "reworded", "imagine", "consider", "infix", "suffix",
            "embedded", "target", "context", "context_long"]
PARAPHRASE_VARIANTS = ["prefix", "reworded", "imagine", "consider"]
AFTER_TEXT_VARIANTS = ["infix", "suffix", "embedded", "target"]   # identity placed after the item text

CHART_VARIANTS = ["prefix", "infix", "suffix"]
CHART_CONDITIONS = ["gender_woman", "religion_muslim", "region_western_europe",
                    "region_arab_culture", "placebo_runner", "placebo_stairs"]


def noun(cond_name):
    return CONDITIONS[cond_name][0]


def identity_sentence(variant, cond_name):
    return SENTENCE_TEMPLATES[variant].format(n=noun(cond_name))


def build_content(variant, cond_name, text):
    """The user-turn content for one (variant, condition) pair."""
    base = PROMPT.format(text=text)
    if cond_name == "baseline":
        return base
    if variant in PARAPHRASE_VARIANTS:
        return f"{identity_sentence(variant, cond_name)} {base}"
    if variant == "suffix":
        return f"{base} {identity_sentence(variant, cond_name)}"
    if variant == "infix":
        return f'Text: "{text}"\n{identity_sentence(variant, cond_name)} {INSTRUCTION_TEXT}'
    if variant == "embedded":
        return f'Text: "{text}"\nAs {noun(cond_name)}, {INSTRUCTION_TEXT_EMBEDDED}'
    if variant == "target":
        return f'Text: "{text}"\n{TARGET_BEFORE}{noun(cond_name)}{TARGET_AFTER}'
    if variant == "context":
        return f"{identity_sentence(variant, cond_name)} {ELABORATION} {base}"
    if variant == "context_long":
        return f"{identity_sentence(variant, cond_name)} {ELABORATION} {PERSPECTIVE} {base}"
    raise ValueError(variant)


def demographic_target(variant, cond_name):
    """The literal string that forms the 'demographic' segment: the identity
    sentence, or just the noun for embedded/target."""
    if variant in ("embedded", "target"):
        return noun(cond_name)
    return identity_sentence(variant, cond_name)


def phrase_of(variant, cond_name):
    return "" if cond_name == "baseline" else demographic_target(variant, cond_name)


def build_prompt_variant(tok, variant, cond_name, text):
    messages = [{"role": "user", "content": build_content(variant, cond_name, text)}]
    return tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def _span_list(span):
    """None / (s,e) / list of (s,e)-or-None -> flat list of resolved tuples.
    "target" has two instruction fragments, every other variant one."""
    if span is None:
        return []
    if isinstance(span, list):
        return [s for s in span if s is not None]
    return [span]


def _span_ok(span):
    if span is None:
        return False
    if isinstance(span, list):
        return len(span) > 0 and all(s is not None for s in span)
    return True


def span_len(span):
    return sum(e - s + 1 for s, e in _span_list(span))


def locate_spans(tokens, variant, cond_name):
    """(demo_span, instr_span, text_span); instr_span is a list of two spans
    for "target", a single span otherwise.

    Anything the template puts AFTER the item text is searched from the end
    (last=True): item texts can contain the same words ("a man", "a Muslim"),
    and a first-occurrence search would then land inside the item text. Spans
    before the item text keep the first occurrence for the same reason."""
    after_text = variant in AFTER_TEXT_VARIANTS
    demo_span = None if cond_name == "baseline" else \
        find_token_span(tokens, demographic_target(variant, cond_name), last=after_text)
    if variant == "target":
        # searched without its trailing space: the space is part of the noun's first
        # token (" someone"), which would otherwise count in both segments
        instr_span = [find_token_span(tokens, TARGET_BEFORE.rstrip(), last=True),
                      find_token_span(tokens, TARGET_AFTER, last=True)]
    else:
        instr_span = find_token_span(tokens, INSTRUCTION_TEXT_EMBEDDED if variant == "embedded"
                                     else INSTRUCTION_TEXT, last=True)
    text_span = find_span_by_delimiters(tokens, TEXT_PREFIX, TEXT_SUFFIX)
    return demo_span, instr_span, text_span


def tokenize_only(tok, prompt_text):
    ids = tok(prompt_text, truncation=True, max_length=MAX_LENGTH,
              add_special_tokens=False)["input_ids"]
    return tok.batch_decode([[tid] for tid in ids])


def spans_problem(spans, variant, cond):
    """None if the located spans are complete and sit where the template puts
    them, else a short reason. The identity span must not overlap the item
    text and must be on the right side of it."""
    demo, instr, txt = spans
    if not _span_ok(instr) or txt is None:
        return "instruction or item text not found"
    if cond != "baseline" and demo is None:
        return "demographic not found"
    covered = [set(range(s, e + 1)) for span in (demo, instr, txt) for s, e in _span_list(span)]
    if sum(len(c) for c in covered) != len(set().union(*covered)):
        return "segments overlap"
    if cond == "baseline":
        return None
    if variant in AFTER_TEXT_VARIANTS:
        return None if demo[0] > txt[1] else "demographic span not after the item text"
    return None if demo[1] < txt[0] else "demographic span not before the item text"


def n_tokens(tok, phrase):
    return len(tok(phrase, add_special_tokens=False)["input_ids"])


_MATCH_SAMPLE_TEXT = "This is an example comment."


def span_length_table(tok):
    """In-context token length of the demographic span for every (variant,
    condition). The span is delimited by fixed template text, so the item text
    does not affect it. Returns DataFrame indexed by condition, one column per
    variant."""
    rows = {}
    for c in CONDITIONS:
        rows[c] = {}
        for v in VARIANTS:
            tokens = tokenize_only(tok, build_prompt_variant(tok, v, c, _MATCH_SAMPLE_TEXT))
            demo, _, _ = locate_spans(tokens, v, c)
            rows[c][v] = -1 if demo is None else span_len(demo)
    return pd.DataFrame.from_dict(rows, orient="index")[VARIANTS]


def build_matching(tok):
    """Every demographic -> the placebos whose span length equals its own in
    EVERY variant. Aborts if a demographic has none."""
    lengths = span_length_table(tok)
    matches, missing = {}, []
    for d in DEMOGRAPHICS:
        m = [p for p in PLACEBOS if (lengths.loc[p] == lengths.loc[d]).all()]
        matches[d] = m
        if not m:
            missing.append(d)
    if missing:
        print(lengths.to_string())
        raise SystemExit(f"No exact-length placebo in every variant for: {missing}. "
                         "Fix the placebo nouns before running.")
    return matches, lengths


def build_run_plan():
    """Full factorial: every condition under every variant."""
    return [(v, c) for v in VARIANTS for c in CONDITIONS]


def preflight(tok, plan, sample_texts):
    """Tokenizer-only dry run over the whole plan on real item texts."""
    failures = []
    for k, (item_id, text) in enumerate(sample_texts):
        for (variant, cond) in plan + [("prefix", "baseline")]:
            tokens = tokenize_only(tok, build_prompt_variant(tok, variant, cond, text))
            why = spans_problem(locate_spans(tokens, variant, cond), variant, cond)
            if why:
                failures.append((item_id, variant, cond, why))
        if (k + 1) % 100 == 0:
            print(f"  preflight {k + 1}/{len(sample_texts)} items", flush=True)
    if failures:
        print(f"\n!!! PREFLIGHT FAILED: {len(failures)} prompt(s) with bad spans for {MODEL_NAME}")
        print(pd.DataFrame(failures, columns=["item_id", "variant", "condition", "problem"])
              .head(40).to_string(index=False))
        raise SystemExit("Aborting before any forward pass.")
    print(f"Preflight OK: spans located and correctly placed for all {len(plan)} (variant, condition) "
          f"pairs + baseline on all {len(sample_texts)} item texts.")


def load_items():
    if not os.path.exists(ITEMS_CSV):
        raise SystemExit(f"{ITEMS_CSV} not found: run select_items.py (score, then select) first.")
    items = pd.read_csv(ITEMS_CSV)
    return items.head(MAX_ITEMS) if MAX_ITEMS else items


class DecisionHead:
    """fp32 logits of the two answer tokens. A bf16 model also rounds its output
    logits to bf16; at typical magnitudes (~25) that is a 0.06-0.125 step in
    log-odds, so small framing shifts get quantized (seen on the test model:
    16 distinct values for 60 prompts). This recomputes just the "0" and "1"
    logits from the final hidden state in fp32. Works for batches; the last
    position must be a real token (left padding)."""

    def __init__(self, model, zero_id, one_id):
        head = model.get_output_embeddings()
        assert getattr(head, "bias", None) is None, "lm_head has a bias; add it here"
        self.w = head.weight[[zero_id, one_id]].detach().float()      # (2, hidden)
        self.h = None
        head.register_forward_hook(lambda mod, inp, out: setattr(self, "h", inp[0][:, -1].detach()))

    def logits(self):
        """(batch, 2) fp32 logits for ["0", "1"] from the last forward pass."""
        return self.h.float() @ self.w.T

    def logodds(self):
        l = self.logits()
        return l[:, 1] - l[:, 0]


class ValueNorms:
    """Captures each layer's value projections during a forward pass and turns
    raw attention at the decision token into value-weighted attention:
    a_hj * ||W_O^h v_j||, renormalized per head (Kobayashi et al. 2020).
    ||W_O^h v||^2 = v^T (W_O^h^T W_O^h) v, so the Gram matrices are precomputed
    once and each pass costs a few small matmuls."""

    def __init__(self, model):
        cfg = model.config
        self.n_heads = cfg.num_attention_heads
        self.n_kv = getattr(cfg, "num_key_value_heads", self.n_heads)
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.n_heads
        self.layers = model.model.layers
        self.values = [None] * len(self.layers)
        self.grams = []
        for li, layer in enumerate(self.layers):
            w = layer.self_attn.o_proj.weight.detach().float()          # (hidden, heads*d)
            w = w.view(w.shape[0], self.n_heads, self.head_dim).permute(1, 2, 0)   # (heads, d, hidden)
            self.grams.append(w @ w.transpose(1, 2))                     # (heads, d, d)
            layer.self_attn.v_proj.register_forward_hook(self._hook(li))

    def _hook(self, li):
        def fn(module, inputs, output):
            self.values[li] = output[0].detach()                         # (seq, kv*d)
        return fn

    def weighted(self, li, attn_last):
        """attn_last: (heads, seq) float tensor -> (heads, seq) value-weighted shares."""
        v = self.values[li].float().view(-1, self.n_kv, self.head_dim).permute(1, 0, 2)  # (kv, seq, d)
        v = v.repeat_interleave(self.n_heads // self.n_kv, dim=0)                       # (heads, seq, d)
        norms = torch.einsum("hsd,hde,hse->hs", v, self.grams[li], v).clamp_min(0).sqrt()
        w = attn_last * norms
        return w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_main = f"{RESULTS_DIR}/master_grid_{MODEL_TAG}.csv.gz"
    out_head = f"{RESULTS_DIR}/master_grid_perhead_{MODEL_TAG}.npz"
    out_match = f"{RESULTS_DIR}/matching_{MODEL_TAG}.csv"
    out_noise = f"{RESULTS_DIR}/noise_{MODEL_TAG}.csv"

    items = load_items()
    print(f"{len(items)} items from {ITEMS_CSV}  split {items.split.value_counts().to_dict()}  "
          f"categories {items.category.value_counts().to_dict()}")

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    matches, lengths = build_matching(tok)
    lengths.to_csv(out_match)
    print(f"\n=== Matched placebos (exact span length in all {len(VARIANTS)} variants) ===")
    for d, ps in matches.items():
        print(f"  {d:32s} {list(lengths.loc[d])}  <-> {', '.join(ps)}")

    plan = build_run_plan()
    print(f"\n=== Run plan: {len(VARIANTS)} variants x {len(CONDITIONS)} conditions = {len(plan)} pairs ===")
    preflight(tok, plan, list(zip(items["item_id"], items["text"])))
    if PREFLIGHT_ONLY:
        print(f"\nPREFLIGHT_ONLY=1: OK. Span lengths written to {out_match}. Exiting before model load.")
        raise SystemExit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading weights, then moving to {device}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        attn_implementation="eager").to(device)
    model.eval()
    vnorm = ValueNorms(model)
    zero_id = tok.encode("0", add_special_tokens=False)[0]
    one_id = tok.encode("1", add_special_tokens=False)[0]
    dhead = DecisionHead(model, zero_id, one_id)

    def run_one(prompt_text):
        """One deterministic forward pass. Returns P(yes), log-odds, raw and
        value-weighted attention at the decision token (layers, heads, seq),
        and the decoded tokens."""
        inputs = tok(prompt_text, return_tensors="pt", truncation=True,
                     max_length=MAX_LENGTH, add_special_tokens=False).to(model.device)
        out = model(**inputs, output_attentions=True)
        logodds = dhead.logodds()[0].item()                 # fp32, see DecisionHead
        prob_one = 1.0 / (1.0 + np.exp(-logodds))
        raw = torch.stack([a[0, :, -1, :].float() for a in out.attentions])          # (L, H, seq)
        vw = torch.stack([vnorm.weighted(li, raw[li]) for li in range(raw.shape[0])])
        tokens = tok.batch_decode([[t] for t in inputs["input_ids"][0].tolist()])
        return prob_one, logodds, raw.cpu().numpy(), vw.cpu().numpy(), tokens

    def segment_attention(attn_lh, demo_span, instr_span, text_span):
        """(layers, heads, seq) -> (layers, heads, 4); 'other' = sink + scaffolding
        (+ the shared elaboration in the context variants)."""
        n_layers, n_heads, _ = attn_lh.shape
        seg = np.zeros((n_layers, n_heads, 4), dtype=np.float32)
        for k, span in enumerate([demo_span, instr_span, text_span]):
            for s, e in _span_list(span):
                seg[:, :, k] += attn_lh[:, :, s: e + 1].sum(axis=2)
        seg[:, :, 3] = 1.0 - seg[:, :, :3].sum(axis=2)
        return seg

    def content_chart_entry(attn, tokens, demo_span, instr_span, text_span):
        spans = [("demographic", demo_span), ("instruction", instr_span), ("item_text", text_span)]
        idx_to_label = {i: label for label, span in spans for s, e in _span_list(span)
                        for i in range(s, e + 1)}
        content_idx = sorted(idx_to_label)
        w = np.array([attn[i] for i in content_idx])
        w = w / w.sum() if w.sum() > 0 else w
        return [tokens[i] for i in content_idx], w, [idx_to_label[i] for i in content_idx]

    result_rows, head_raw, head_vw, head_index = [], [], [], []

    def summarize(seg):
        d, i_, t, o = [float(x) for x in seg.mean(axis=(0, 1))]
        content = 1.0 - o
        norm = (lambda x: x / content) if content > 0 else (lambda x: np.nan)
        return d, i_, t, o, norm(d), norm(i_), norm(t)

    def record(item, variant, cond, prob, logodds, base, seg_raw, seg_vw, spans, seq_len):
        demo_span, instr_span, text_span = spans
        noun_, axis, is_placebo = CONDITIONS[cond] if cond != "baseline" else ("", "baseline", False)
        phrase = phrase_of(variant, cond)
        row = {
            "model": MODEL_TAG, "item_id": item["item_id"], "split": item["split"],
            "category": item["category"], "sub_category": item["sub_category"],
            "entropy": item["entropy"], "variance": item["variance"],
            "p_offensive_raters": item["p_offensive"], "n_raters": item["n_raters"],
            "variant": variant, "condition": cond, "axis": axis, "is_placebo": is_placebo,
            "noun": noun_, "phrase": phrase,
            "matched_placebos": "|".join(matches.get(cond, [])),
            "phrase_tokens": n_tokens(tok, phrase) if phrase else 0,
            "demo_span_tokens": span_len(demo_span), "instr_span_tokens": span_len(instr_span),
            "text_span_tokens": span_len(text_span), "seq_len": seq_len,
            "predicted": int(logodds > 0), "prob_yes": prob, "logodds_yes": logodds,
            "baseline_pred": int(base[1] > 0), "baseline_prob_yes": base[0],
            "baseline_logodds_yes": base[1],
            "flipped_vs_baseline": int(logodds > 0) != int(base[1] > 0),
            "delta_prob_yes": prob - base[0], "delta_logodds_yes": logodds - base[1],
        }
        for tag, seg in (("", seg_raw), ("_vw", seg_vw)):
            d, i_, t, o, dn, in_, tn = summarize(seg)
            row.update({f"pct_demographic{tag}": d, f"pct_instruction{tag}": i_,
                        f"pct_item_text{tag}": t, f"pct_other_scaffolding{tag}": o,
                        f"pct_demographic_norm{tag}": dn, f"pct_instruction_norm{tag}": in_,
                        f"pct_item_text_norm{tag}": tn})
        result_rows.append(row)
        head_raw.append(seg_raw.astype(np.float16))
        head_vw.append(seg_vw.astype(np.float16))
        head_index.append((item["item_id"], variant, cond, span_len(demo_span),
                           span_len(instr_span), span_len(text_span), seq_len))

    def run_pair(item, variant, cond, base):
        prompt = build_prompt_variant(tok, variant, cond, item["text"])
        prob, logodds, raw, vw, tokens = run_one(prompt)
        spans = locate_spans(tokens, variant, cond)
        why = spans_problem(spans, variant, cond)
        if why:
            raise SystemExit(f"item {item['item_id']} {variant} {cond}: {why}. Stopping instead "
                             "of writing wrong attention values.")
        return prob, logodds, raw, vw, tokens, spans

    # --- stage 1: baselines --------------------------------------------------
    print(f"\n=== Stage 1: baseline pass over {len(items)} items ===")
    baselines = {}
    t_start = time.time()
    with torch.no_grad():
        for i, item in items.iterrows():
            prob, logodds, raw, vw, tokens, spans = run_pair(item, "prefix", "baseline", None)
            baselines[item["item_id"]] = (prob, logodds)
            record(item, "none", "baseline", prob, logodds, (prob, logodds),
                   segment_attention(raw, *spans), segment_attention(vw, *spans), spans, len(tokens))
    print(f"  done in {(time.time() - t_start) / 60:.1f} min")

    # illustrative items: top-N by rating entropy in the shared list, so both
    # models chart the same items
    illustrative = items.sort_values("entropy", ascending=False).head(N_ILLUSTRATIVE)
    chart_data = {}
    with torch.no_grad():
        for _, item in illustrative.iterrows():
            _, _, raw, _, tokens, spans = run_pair(item, "prefix", "baseline", None)
            entry = content_chart_entry(raw.mean(axis=(0, 1)), tokens, *spans)
            chart_data[item["item_id"]] = {v: {"baseline": entry} for v in CHART_VARIANTS}

    # --- stage 2: the full grid ---------------------------------------------
    n_total = len(items) * len(plan)
    print(f"\n=== Stage 2: {len(plan)} pairs x {len(items)} items = {n_total} forward passes ===")
    n_done, t_start = 0, time.time()
    with torch.no_grad():
        for _, item in items.iterrows():
            base = baselines[item["item_id"]]
            for (variant, cond) in plan:
                prob, logodds, raw, vw, tokens, spans = run_pair(item, variant, cond, base)
                record(item, variant, cond, prob, logodds, base,
                       segment_attention(raw, *spans), segment_attention(vw, *spans), spans, len(tokens))
                if item["item_id"] in chart_data and variant in CHART_VARIANTS and cond in CHART_CONDITIONS:
                    chart_data[item["item_id"]][variant][cond] = content_chart_entry(
                        raw.mean(axis=(0, 1)), tokens, *spans)
                n_done += 1
                if n_done % 1000 == 0:
                    el = time.time() - t_start
                    rate = n_done / el
                    print(f"  {n_done}/{n_total}  elapsed {el / 60:.1f} min  {rate:.1f} passes/s  "
                          f"ETA {(n_total - n_done) / rate / 60:.1f} min", flush=True)
                if n_done % CHECKPOINT_EVERY == 0:
                    pd.DataFrame(result_rows).to_csv(out_main + ".partial", index=False)

    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(out_main, index=False, compression="gzip")
    if os.path.exists(out_main + ".partial"):
        os.remove(out_main + ".partial")
    print(f"\nSaved {out_main}  ({len(results_df)} rows)")

    idx = pd.DataFrame(head_index, columns=["item_id", "variant", "condition", "demo_span_tokens",
                                            "instr_span_tokens", "text_span_tokens", "seq_len"])
    head_raw_arr, head_vw_arr = np.stack(head_raw), np.stack(head_vw)
    np.savez_compressed(
        out_head, attention=head_raw_arr, attention_vw=head_vw_arr,
        item_id=idx["item_id"].to_numpy(dtype=np.int64),
        variant=np.asarray(idx["variant"].tolist(), dtype=str),
        condition=np.asarray(idx["condition"].tolist(), dtype=str),
        demo_span_tokens=idx["demo_span_tokens"].to_numpy(dtype=np.int32),
        instr_span_tokens=idx["instr_span_tokens"].to_numpy(dtype=np.int32),
        text_span_tokens=idx["text_span_tokens"].to_numpy(dtype=np.int32),
        seq_len=idx["seq_len"].to_numpy(dtype=np.int32), segments=np.array(SEGMENTS))
    print(f"Saved {out_head}  shape={head_raw_arr.shape} x2 (raw, value-weighted), float16")

    # --- output sanity checks -------------------------------------------------
    expected = len(items) * (len(plan) + 1)
    seg_sum = head_raw_arr[..., :3].astype(np.float32).sum(-1)
    print(f"\n=== Sanity checks ===")
    print(f"  rows {len(results_df)} (expected {expected}) -> {'OK' if len(results_df) == expected else 'MISMATCH'}")
    print(f"  NaN in attention columns: {int(results_df.filter(like='pct_').isna().sum().sum())}")
    print(f"  per-head content share in [0, 1]: min {seg_sum.min():.4f} max {seg_sum.max():.4f}")
    span_diff = results_df[results_df.condition != "baseline"].groupby(["variant", "condition"]) \
        ["demo_span_tokens"].nunique()
    print(f"  (variant, condition) pairs whose span length varies across items: {(span_diff > 1).sum()}")

    # --- noise check ----------------------------------------------------------
    if NOISE_ITEMS:
        print(f"\n=== Noise check: first {NOISE_ITEMS} items again ===")
        ref = results_df.set_index(["item_id", "variant", "condition"])
        noise = []
        with torch.no_grad():
            for _, item in items.head(NOISE_ITEMS).iterrows():
                for (variant, cond) in [("none", "baseline")] + plan:
                    v = "prefix" if cond == "baseline" else variant
                    prob, logodds, *_ = run_pair(item, v, cond, None)
                    r = ref.loc[(item["item_id"], variant, cond)]
                    noise.append({"item_id": item["item_id"], "variant": variant, "condition": cond,
                                  "d_prob": prob - r["prob_yes"], "d_logodds": logodds - r["logodds_yes"]})
        noise = pd.DataFrame(noise)
        score_path = f"{SCORES_DIR}/scores_{MODEL_TAG}.csv"
        if os.path.exists(score_path):
            sc = pd.read_csv(score_path).set_index("item_id")
            b = results_df[results_df.condition == "baseline"].set_index("item_id")
            d = b["prob_yes"] - sc.loc[b.index, "prob_yes"]
            print(f"  grid baseline vs select_items score (no output_attentions): "
                  f"max |dP| {d.abs().max():.2e}, mean |dP| {d.abs().mean():.2e}")
        noise.to_csv(out_noise, index=False)
        print(f"  identical input run twice: max |dP| {noise.d_prob.abs().max():.2e}, "
              f"max |d logodds| {noise.d_logodds.abs().max():.2e}  ({len(noise)} passes) -> {out_noise}")

    grid = results_df[results_df.condition != "baseline"]
    pd.set_option("display.width", 250)
    print("\n=== Mean signed delta P(yes) by condition x variant ===")
    print(grid.pivot_table(index="condition", columns="variant", values="delta_prob_yes")
          .round(3)[VARIANTS].to_string())

    # --- illustrative content-attention charts: prefix vs infix vs suffix -----
    charts_dir = f"{FIGURES_DIR}/content_attention_charts"
    os.makedirs(charts_dir, exist_ok=True)
    n_saved = 0
    for item_id, per_variant in chart_data.items():
        for variant in CHART_VARIANTS:
            conds_present = [c for c in ["baseline"] + CHART_CONDITIONS if c in per_variant[variant]]
            fig, axes = plt.subplots(len(conds_present), 1, figsize=(12, 3 * len(conds_present)), squeeze=False)
            for ax, cond_name in zip(axes[:, 0], conds_present):
                toks, weights, labels = per_variant[variant][cond_name]
                ax.bar(range(len(toks)), weights, color=[SEGMENT_COLORS[l] for l in labels])
                ax.set_xticks(range(len(toks)))
                ax.set_xticklabels([t.strip() or "·" for t in toks], rotation=60, ha="right", fontsize=8)
                ax.set_title(cond_name)
                ax.set_ylabel("attention\n(sink excluded, renorm.)")
            handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SEGMENT_COLORS.values()]
            fig.legend(handles, SEGMENT_COLORS.keys(), loc="upper right")
            fig.suptitle(f"model={MODEL_TAG}  item_id={item_id}  variant={variant}")
            fig.tight_layout()
            fig.savefig(f"{charts_dir}/{MODEL_TAG}_item{item_id}_{variant}.png", dpi=150)
            plt.close(fig)
            n_saved += 1
    print(f"\nSaved {n_saved} illustrative charts to {charts_dir}/")
