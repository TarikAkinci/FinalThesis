"""
Consolidated grid run: one job that fills every column of the master table.

Design is BLOCK-based rather than one full conditions x variants factorial.
A full factorial over 17 conditions x 8 structures x 500 items is ~68k forward
passes, most of them redundant -- the breadth question ("do the 8 D3CODE
regions differ?") does not need all 8 prompt structures to answer, and the
structure question does not need all 17 conditions. Each block below is a
sub-grid aimed at one question, and the run plan is their deduplicated union.

  BREADTH    all 8 D3CODE regions + gender + religion + bare identity +
             3 placebos, under `prefix` only.
             Answers: is "Arab Culture" actually special, or does every
             non-Western region behave the same way? Previously only Western
             Europe vs Arab Culture were tested, which cannot distinguish
             "Arab is special" from "anything not Western Europe is special".

  STRUCTURE  the 5 conditions carried over from the previous study, under
             suffix/reworded/embedded. Keeps continuity with the existing
             table (their `prefix` rows come from BREADTH).

  MATCHED    EXACT token-matched demographic/placebo pairs under all 4
             structures. Word-count matching was the previous approximation;
             this matches real tokenizer output. For each selected
             demographic phrase we find a content-free placebo from
             PLACEBO_POOL with an identical token count, so any remaining
             difference cannot be attributed to prompt length. Matching runs
             at runtime against whichever model's tokenizer is loaded, and
             the resulting pairing is printed, because Llama and Qwen
             tokenize these phrases differently.

  CONTEXT    elaborated framings: the demographic phrase plus a sentence of
             context, and a longer version that also instructs the model to
             answer from that perspective. Placebos get a length-matched,
             content-free elaboration so this block is not simply measuring
             "longer prompt".

Item pool: see ITEM SELECTION below -- the in-band subset of the seeded
500-item D3CODE eval sample, with entropy demoted from a selection filter to
a recorded covariate.

Per-layer and per-head attention is recorded for every row. This costs no
extra forward passes: the same output_attentions call already contains it,
the earlier scripts just averaged it away before writing.

Model-parameterized so the identical grid runs on Qwen2.5-7B for a
model-comparison axis. A tokenizer-only preflight aborts before the model
reaches the GPU if any span is unlocatable under that model's chat template.
"""
import os
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use("Agg")   # imported at module load, not inside __main__, so a missing
import matplotlib.pyplot as plt   # matplotlib install shows up immediately at script
                                   # start (like the earlier sklearn issue), not mid-run

# torch >= 2.13 routes some eager CUDA ops (e.g. the RoPE outer-product bmm, RMSNorm, topk)
# through experimental Triton / CuTe-DSL kernels. On the LRZ nodes the first such call makes
# Triton compile a small C helper with gcc, which fails there, and every forward pass dies
# (see the smoke-test logs). Switching the overrides off sends those ops back to PyTorch's
# regular cuBLAS/ATen path. Numerically equivalent, and needs no compiler on the node.
# Guarded: on a torch without these modules this is a no-op rather than an error.
for _mod in ("triton_utils", "cutedsl_utils"):
    try:
        getattr(__import__("torch._native", fromlist=[_mod]), _mod).deregister_op_overrides()
        print(f"torch native {_mod} op overrides disabled", flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f"(torch native {_mod} override disable skipped: {_e!r})", flush=True)

from span_utils import find_token_span, find_span_by_delimiters
from zeroshot_two_datasets import PROMPT
from high_disagreement_items import rank_by_entropy

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
MODEL_TAG = os.environ.get("MODEL_TAG", MODEL_NAME.split("/")[-1].replace(".", "-"))
MAX_LENGTH = 1024
RESULTS_DIR = "results/master_grid"

# ITEM SELECTION -------------------------------------------------------------
# Baseline P(yes) must fall in this band. Items pinned near 0 or 1 at baseline
# have no room for any perturbation to move the prediction further in that
# direction, so their delta is mechanically compressed toward zero regardless
# of what the prompt says. This was a real, diagnosed confound in the earlier
# entropy-only selection and the filter stays.
MIN_BASELINE_PROB = 0.02
MAX_BASELINE_PROB = 0.98
# Entropy is NO LONGER a selection criterion -- it is written to every row as a
# covariate instead. Selecting on disagreement up front both shrinks the sample
# and bakes in a selection effect; keeping it as a column lets the analysis ask
# "does rater disagreement predict prompt sensitivity?" as a regression across
# the full in-band range, which is strictly more informative than pre-filtering.
# Set MAX_ITEMS to cap the run for a smoke test; 0 means no cap.
MAX_ITEMS = int(os.environ.get("MAX_ITEMS", "0"))
# A capped run is a smoke test: write it to a separate directory so a leftover 3-item
# file can never sit next to (or be mistaken for) a real result.
if MAX_ITEMS:
    RESULTS_DIR = RESULTS_DIR + "_smoke"
# Exit right after the tokenizer-only preflight, before the model loads. Runs on a CPU
# node: checks imports, data, item ranking, token matching and span-finding for this
# model's chat template without occupying a GPU.
PREFLIGHT_ONLY = os.environ.get("PREFLIGHT_ONLY", "0") == "1"

# Illustrative content-attention charts, prefix vs infix vs suffix, same phrase per
# condition (so token length is identical across the three, verified locally -- see
# conversation), to show the position effect visually alongside the aggregate numbers.
FIGURES_DIR = "figures/master_grid"
N_ILLUSTRATIVE = 3
CHART_VARIANTS = ["prefix", "infix", "suffix"]
SEGMENT_COLORS = {"demographic": "tab:red", "instruction": "tab:blue", "item_text": "tab:green"}
CHECKPOINT_EVERY = 1000   # grid passes between partial saves of the main CSV

SEGMENTS = ["demographic", "instruction", "item_text", "other"]

INSTRUCTION_TEXT = "Is this text offensive? Answer with only 0 (no) or 1 (yes)."
INSTRUCTION_TEXT_EMBEDDED = "is this text offensive? Answer with only 0 (no) or 1 (yes)."
# "target" variant: asks whether the text is offensive FOR/TO a group, rather than
# having the model self-identify AS that group ("As a woman, is this offensive?"). A
# semantically distinct manipulation, not a cosmetic one. The noun sits inside the
# question, splitting the instruction into a before- and after-fragment around it.
TARGET_BEFORE = "Do you think this text is offensive for "
TARGET_AFTER = "? Answer with only 0 (no) or 1 (yes)."
TEXT_PREFIX, _ = PROMPT.split("{text}")
TEXT_SUFFIX = '"\n'

# Elaborations for the CONTEXT block. Deliberately parallel in structure and
# equal in word count so the demographic and placebo versions differ in content
# only, not in how much text was added.
ELABORATION_DEMO = "Your background, values, and everyday experience shape what you find offensive."
ELABORATION_PLACEBO = "Your background, habits, and everyday routine shape what you find refreshing."
PERSPECTIVE = "Answer from that perspective."


def _reword(phrase):
    """Uniform +1-word transformation, applied identically to every phrase.
    Kept mechanical so the reworded variant can never drift back into the
    length asymmetry that an earlier hand-typed version had."""
    assert phrase.startswith("You ")
    return "Assume you " + phrase[4:]


# --- conditions -------------------------------------------------------------
# (phrase, noun_phrase_for_embedded, axis, is_placebo)
# Region names are D3CODE's own Region values, verbatim, all 8 of them.
def _region(key, display, noun_display=None):
    noun = noun_display or display
    return (f"region_{key}", (f"You are someone from {display}.",
                              f"someone from {noun}", "region", False))


CONDITIONS = dict([
    _region("western_europe", "Western Europe"),
    _region("north_america", "North America"),
    _region("latin_america", "Latin America"),
    _region("arab_culture", "the Arab Culture region"),
    _region("indian_cultural_sphere", "the Indian Cultural Sphere"),
    _region("sinosphere", "the Sinosphere"),
    _region("sub_saharan_africa", "Sub Saharan Africa"),
    _region("oceania", "Oceania"),
    ("gender_man", ("You are a man.", "a man", "gender", False)),
    ("gender_woman", ("You are a woman.", "a woman", "gender", False)),
    # wording carried over verbatim from attention_probe.py's CONDITIONS
    ("religion_christian", ("You are Christian.", "a Christian", "religion", False)),
    ("religion_muslim", ("You are Muslim.", "a Muslim", "religion", False)),
    # bare identity, to separate "Arab" the identity from the long region phrase
    ("identity_arab", ("You are Arab.", "an Arab", "identity", False)),
    ("identity_european", ("You are European.", "a European", "identity", False)),
    # core placebos, kept as fixed phrases. placebo_long is the exact sentence
    # every previous study used, so those results stay comparable.
    ("placebo_short", ("You are left-handed.", "someone left-handed", "placebo", True)),
    ("placebo_mid", ("You are a swimmer.", "a swimmer", "placebo", True)),
    ("placebo_long", ("You are someone who prefers tea over coffee.",
                      "someone who prefers tea over coffee", "placebo", True)),
])

# Candidate placebos for EXACT token matching in the MATCHED block. Spans a
# range of lengths so a match exists for each demographic phrase under either
# model's tokenizer. All are content-free with respect to the task: none names
# a group, an origin, a belief, or anything a rater demographic could encode.
PLACEBO_POOL = [
    # Deliberately dense across the 4-14 token range, with several phrases per
    # length, because an EXACT match has to exist under whichever tokenizer is
    # loaded and Llama and Qwen split these differently. All are content-free
    # with respect to the task: none names a group, an origin, a belief, or
    # anything a D3CODE rater demographic could encode.
    ("You are tall.", "someone tall"),
    ("You are calm.", "someone calm"),
    ("You are punctual.", "someone punctual"),
    ("You are a runner.", "a runner"),
    ("You are left-handed.", "someone left-handed"),
    ("You are a swimmer.", "a swimmer"),
    ("You are a morning person.", "a morning person"),
    ("You are an occasional cyclist.", "an occasional cyclist"),
    ("You are an early riser.", "an early riser"),
    ("You are a frequent walker.", "a frequent walker"),
    ("You are someone who bakes.", "someone who bakes"),
    ("You are someone who prefers tea.", "someone who prefers tea"),
    ("You are someone who collects old maps.", "someone who collects old maps"),
    ("You are someone who enjoys crossword puzzles.", "someone who enjoys crossword puzzles"),
    ("You are someone who prefers tea over coffee.", "someone who prefers tea over coffee"),
    ("You are someone who keeps a tidy desk drawer.", "someone who keeps a tidy desk drawer"),
    ("You are someone who usually takes the stairs instead.", "someone who usually takes the stairs instead"),
    ("You are someone who prefers window seats on long flights.",
     "someone who prefers window seats on long flights"),
    ("You are someone who waters the office plants every other week.",
     "someone who waters the office plants every other week"),
    ("You are someone who always carries a spare umbrella in their bag.",
     "someone who always carries a spare umbrella in their bag"),
]

# Demographic phrases to build exact token-matched placebo pairs for. Chosen to
# span the full length range and to include the two conditions the earlier
# results hinged on (Western Europe and Arab Culture).
MATCHED_DEMOGRAPHICS = ["religion_muslim", "gender_woman",
                        "region_western_europe", "region_arab_culture"]

STRUCTURE_CONDITIONS = ["gender_man", "gender_woman", "region_western_europe",
                        "region_arab_culture", "placebo_long"]
# same 5 conditions, reused as which rows the illustrative content-attention
# charts show (defined here, after STRUCTURE_CONDITIONS exists to alias)
CHART_CONDITIONS = STRUCTURE_CONDITIONS
CONTEXT_CONDITIONS = ["gender_woman", "region_western_europe", "region_arab_culture",
                      "region_sinosphere", "placebo_mid", "placebo_long"]

STRUCTURE_VARIANTS = ["prefix", "suffix", "reworded", "embedded"]
TARGET_VARIANTS = ["target"]
INFIX_VARIANTS = ["infix"]
CONTEXT_VARIANTS = ["context", "context_long"]


def build_content(variant, cond_name, text):
    """The user-turn content for one (variant, condition) pair."""
    base = PROMPT.format(text=text)
    if cond_name == "baseline":
        return base
    phrase, noun, axis, is_placebo = CONDITIONS[cond_name]
    if variant == "prefix":
        return f"{phrase} {base}"
    if variant == "suffix":
        return f"{base} {phrase}"
    if variant == "infix":
        return f'Text: "{text}"\n{phrase} {INSTRUCTION_TEXT}'
    if variant == "reworded":
        return f"{_reword(phrase)} {base}"
    if variant == "embedded":
        return f'Text: "{text}"\nAs {noun}, {INSTRUCTION_TEXT_EMBEDDED}'
    if variant == "target":
        return f'Text: "{text}"\n{TARGET_BEFORE}{noun}{TARGET_AFTER}'
    if variant == "context":
        return f"{phrase} {_elaboration(is_placebo)} {base}"
    if variant == "context_long":
        return f"{phrase} {_elaboration(is_placebo)} {PERSPECTIVE} {base}"
    raise ValueError(variant)


def _elaboration(is_placebo):
    return ELABORATION_PLACEBO if is_placebo else ELABORATION_DEMO


def demographic_target(variant, cond_name):
    """The exact literal string that constitutes the 'demographic' segment for
    this (variant, condition). For the context variants this deliberately
    includes the elaboration: the whole injected block is what we are measuring
    attention to, not just its first sentence."""
    phrase, noun, axis, is_placebo = CONDITIONS[cond_name]
    if variant in ("embedded", "target"):
        return noun
    if variant == "reworded":
        return _reword(phrase)
    if variant == "context":
        return f"{phrase} {_elaboration(is_placebo)}"
    if variant == "context_long":
        return f"{phrase} {_elaboration(is_placebo)} {PERSPECTIVE}"
    return phrase


def build_prompt_variant(tok, variant, cond_name, text):
    messages = [{"role": "user", "content": build_content(variant, cond_name, text)}]
    return tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def _span_list(span):
    """Normalize a span -- None, a single (start,end) tuple, or a list of those
    (possibly containing unresolved Nones) -- into a flat list of resolved tuples.
    Used because "target" needs two disjoint instruction fragments where every
    other variant needs only one contiguous span."""
    if span is None:
        return []
    if isinstance(span, list):
        return [s for s in span if s is not None]
    return [span]


def _span_ok(span):
    """True if a span (any of the three representations above) is fully
    resolved -- for a list, every sub-span must be found, not just one."""
    if span is None:
        return False
    if isinstance(span, list):
        return len(span) > 0 and all(s is not None for s in span)
    return True


def locate_spans(tokens, variant, cond_name):
    """Returns (demo_span, instr_span, text_span). demo_span/text_span are a
    single (start,end) tuple or None. instr_span is a single tuple for every
    variant except "target", where it is a list of two tuples/Nones (the
    instruction fragment before and after the embedded noun phrase). Use
    _span_list()/_span_ok() rather than checking `is not None` directly, since
    that only works for the single-tuple case.
    """
    if cond_name == "baseline":
        demo_span = None
    else:
        demo_span = find_token_span(tokens, demographic_target(variant, cond_name))
    if variant == "target":
        instr_span = [find_token_span(tokens, TARGET_BEFORE),
                      find_token_span(tokens, TARGET_AFTER)]
    else:
        instr = INSTRUCTION_TEXT_EMBEDDED if variant == "embedded" else INSTRUCTION_TEXT
        instr_span = find_token_span(tokens, instr)
    text_span = find_span_by_delimiters(tokens, TEXT_PREFIX, TEXT_SUFFIX)
    return demo_span, instr_span, text_span


def n_tokens(tok, phrase):
    return len(tok(phrase, add_special_tokens=False)["input_ids"])


def build_matched_conditions(tok):
    """Register one exact token-matched placebo per distinct token length among
    MATCHED_DEMOGRAPHICS, named placebo_tok{N}. Two demographics of the same
    token length share a placebo, which is intended -- it is the same control.
    Returns (matched_condition_names, pairing_rows) for logging."""
    pool_lens = {}
    for phrase, noun in PLACEBO_POOL:
        pool_lens.setdefault(n_tokens(tok, phrase), (phrase, noun))

    matched_names, pairing = [], []
    for demo in MATCHED_DEMOGRAPHICS:
        phrase = CONDITIONS[demo][0]
        tlen = n_tokens(tok, phrase)
        exact = pool_lens.get(tlen)
        if exact is None:
            nearest = min(pool_lens, key=lambda k: abs(k - tlen))
            pairing.append((demo, tlen, pool_lens[nearest][0], nearest, "NEAREST"))
            chosen, clen = pool_lens[nearest], nearest
        else:
            pairing.append((demo, tlen, exact[0], tlen, "exact"))
            chosen, clen = exact, tlen
        name = f"placebo_tok{clen}"
        if name not in CONDITIONS:
            CONDITIONS[name] = (chosen[0], chosen[1], "placebo", True)
        matched_names.append(name)
    return sorted(set(matched_names)), pairing


def build_run_plan(matched_placebos):
    """Deduplicated union of all four blocks.
    Returns {(variant, condition): set-of-block-names}."""
    plan = {}

    def add(variants, conditions, block):
        for v in variants:
            for c in conditions:
                plan.setdefault((v, c), set()).add(block)

    breadth = [c for c, (_, _, axis, _) in CONDITIONS.items()
               if axis in ("region", "gender", "religion", "identity")
               or c in ("placebo_short", "placebo_mid", "placebo_long")]
    add(["prefix"], breadth, "breadth")
    add(["suffix", "reworded", "embedded"] + TARGET_VARIANTS + INFIX_VARIANTS, STRUCTURE_CONDITIONS, "structure")
    add(STRUCTURE_VARIANTS + TARGET_VARIANTS + INFIX_VARIANTS, MATCHED_DEMOGRAPHICS + matched_placebos, "matched")
    add(CONTEXT_VARIANTS, CONTEXT_CONDITIONS, "context")
    return plan


def tokenize_only(tok, prompt_text):
    ids = tok(prompt_text, truncation=True, max_length=MAX_LENGTH,
              add_special_tokens=False)["input_ids"]
    return [tok.decode([tid]) for tid in ids]


def preflight(tok, plan, sample_texts):
    """Tokenizer-only dry run over the whole run plan. Span-finding depends on
    the chat template, which differs between Llama and Qwen, and a silent span
    failure only shows up as NaN columns after the entire job has run. This
    refuses to start the real loop instead. Costs no GPU time."""
    failures = []
    for text in sample_texts:
        for (variant, cond) in sorted(plan):
            tokens = tokenize_only(tok, build_prompt_variant(tok, variant, cond, text))
            demo, instr, txt = locate_spans(tokens, variant, cond)
            need = [("instruction", instr), ("item_text", txt)]
            if cond != "baseline":
                need.append(("demographic", demo))
            for seg_name, span in need:
                if not _span_ok(span):
                    failures.append((variant, cond, seg_name))
    unique = sorted(set(failures))
    if unique:
        print(f"\n!!! PREFLIGHT FAILED: {len(unique)} unlocatable span(s) for {MODEL_NAME}")
        for variant, cond, seg in unique:
            print(f"    variant={variant:13s} condition={cond:32s} segment={seg}")
        raise SystemExit(
            "Aborting before any forward pass. Span-finding must be fixed for this "
            "model's chat template first, otherwise the run produces NaN attention "
            "columns for these combinations and the GPU time is wasted."
        )
    print(f"Preflight OK: all spans located for all {len(plan)} (variant, condition) "
          f"pairs on {len(sample_texts)} sample items.")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_main = f"{RESULTS_DIR}/master_grid_{MODEL_TAG}.csv"
    out_layer = f"{RESULTS_DIR}/master_grid_layerwise_{MODEL_TAG}.csv.gz"
    out_head = f"{RESULTS_DIR}/master_grid_perhead_{MODEL_TAG}.npz"
    out_items = f"{RESULTS_DIR}/master_grid_items_{MODEL_TAG}.csv"

    candidates = rank_by_entropy()
    print(f"{len(candidates)} items in the seeded 500-item D3CODE eval sample "
          f"(entropy {candidates.entropy.min():.3f}-{candidates.entropy.max():.3f})")

    print(f"\nLoading model: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    matched_placebos, pairing = build_matched_conditions(tok)
    print(f"\n=== Exact token-matched pairs (this model's tokenizer) ===")
    for demo, dlen, pphrase, plen, status in pairing:
        flag = "" if status == "exact" else "   <-- NO EXACT MATCH, nearest used"
        print(f"  {demo:26s} {dlen:2d} tok  <->  {plen:2d} tok  \"{pphrase}\"{flag}")

    plan = build_run_plan(matched_placebos)
    by_block = {}
    for pair, blocks in plan.items():
        for b in blocks:
            by_block.setdefault(b, []).append(pair)
    print(f"\n=== Run plan: {len(plan)} unique (variant, condition) pairs ===")
    for b in ["breadth", "structure", "matched", "context"]:
        print(f"  {b:10s} {len(by_block.get(b, []))} pairs")

    preflight(tok, plan, candidates["text"].head(3).tolist())
    if PREFLIGHT_ONLY:
        print("\nPREFLIGHT_ONLY=1: imports, data, token matching and span-finding all OK. "
              "Exiting before model load.")
        raise SystemExit(0)

    # No device_map="auto": that needs the `accelerate` package, which isn't installed on
    # the LRZ venv, and this is a single-GPU job anyway.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading weights, then moving to {device}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device)
    model.eval()

    zero_id = tok.encode("0", add_special_tokens=False)[0]
    one_id = tok.encode("1", add_special_tokens=False)[0]

    def run_one(prompt_text):
        """One deterministic forward pass. Returns per-head attention at the
        decision token, shape (n_layers, n_heads, seq_len) -- not averaged over
        layers/heads here, so the per-head detail survives to disk."""
        inputs = tok(prompt_text, return_tensors="pt", truncation=True,
                     max_length=MAX_LENGTH, add_special_tokens=False).to(model.device)
        out = model(**inputs, output_attentions=True)
        logits = out.logits[0, -1]
        predicted = "1" if logits[one_id] > logits[zero_id] else "0"
        _, prob_one = torch.softmax(logits[[zero_id, one_id]].float(), dim=0)
        attn_stack = torch.stack(out.attentions, dim=0)
        attn_lh = attn_stack[:, 0, :, -1, :].float().cpu().numpy()
        tokens = [tok.decode([t]) for t in inputs["input_ids"][0].tolist()]
        return predicted, prob_one.item(), attn_lh, tokens

    def content_chart_entry(attn, tokens, demo_span, instr_span, text_span):
        """attn: (seq,) -- layer+head-averaged attention at the decision token.
        Returns (tokens, weights, labels) for one condition's content-attention bar
        chart row: sink-excluded, renormalized attention restricted to just the
        demographic/instruction/item-text tokens."""
        spans = [("demographic", demo_span), ("instruction", instr_span), ("item_text", text_span)]
        content_idx = sorted({i for _, span in spans for s, e in _span_list(span) for i in range(s, e + 1)})
        idx_to_label = {i: label for label, span in spans for s, e in _span_list(span) for i in range(s, e + 1)}
        content_attn = np.array([attn[i] for i in content_idx])
        content_attn_norm = content_attn / content_attn.sum() if content_attn.sum() > 0 else content_attn
        return [tokens[i] for i in content_idx], content_attn_norm, [idx_to_label[i] for i in content_idx]


    def segment_attention(attn_lh, demo_span, instr_span, text_span):
        """attn_lh: (n_layers, n_heads, seq) -> (n_layers, n_heads, 4).
        'other' is the remainder: attention sink plus chat-template scaffolding.
        Each of demo_span/instr_span/text_span may be a single (start,end) tuple,
        None, or (instr_span only, for "target") a list of tuples -- summed via
        _span_list() so a segment split across disjoint fragments still totals
        correctly."""
        n_layers, n_heads, _ = attn_lh.shape
        seg = np.zeros((n_layers, n_heads, 4), dtype=np.float32)
        for k, span in enumerate([demo_span, instr_span, text_span]):
            for s, e in _span_list(span):
                seg[:, :, k] += attn_lh[:, :, s: e + 1].sum(axis=2)
        seg[:, :, 3] = 1.0 - seg[:, :, :3].sum(axis=2)
        return seg

    # --- stage 1: baseline pass over the whole seeded sample ----------------
    # Every item gets a baseline row, including the ones that later fall out of
    # band, so the table can still report baseline behaviour on the full 500.
    print(f"\n=== Stage 1: baseline pass over all {len(candidates)} items ===")
    result_rows, layer_rows, head_blocks, head_index = [], [], [], []
    baselines = {}

    def record(item, variant, cond_name, blocks, predicted, prob_yes,
               b_pred, b_prob, seg, ok):
        if ok:
            per_layer = seg.mean(axis=1)
            d, i_, t, o = [float(x) for x in per_layer.mean(axis=0)]
        else:
            per_layer = None
            d = i_ = t = o = np.nan
        content = 1.0 - o if not np.isnan(o) else np.nan
        has = isinstance(content, float) and content > 0
        phrase, noun, axis, is_placebo = (
            CONDITIONS[cond_name] if cond_name != "baseline" else ("", "", "baseline", False))
        result_rows.append({
            "model": MODEL_TAG, "item_id": item["item_id"], "category": item["category"],
            "entropy": item["entropy"], "variance": item["variance"],
            "p_offensive_raters": item["p_offensive"], "n_raters": item["n_raters"],
            "block": "|".join(sorted(blocks)), "variant": variant, "condition": cond_name,
            "axis": axis, "is_placebo": is_placebo,
            "phrase": phrase, "phrase_words": len(phrase.split()) if phrase else 0,
            "phrase_tokens": n_tokens(tok, phrase) if phrase else 0,
            "predicted": predicted, "prob_yes": prob_yes,
            "baseline_pred": b_pred, "baseline_prob_yes": b_prob,
            "flipped_vs_baseline": predicted != b_pred,
            "delta_prob_yes": prob_yes - b_prob,
            "pct_demographic": d, "pct_instruction": i_, "pct_item_text": t,
            "pct_other_scaffolding": o,
            "pct_demographic_norm": d / content if has else np.nan,
            "pct_instruction_norm": i_ / content if has else np.nan,
            "pct_item_text_norm": t / content if has else np.nan,
        })
        if per_layer is not None:
            for li in range(per_layer.shape[0]):
                a, b_, c_, e = [float(x) for x in per_layer[li]]
                layer_rows.append({
                    "model": MODEL_TAG, "item_id": item["item_id"], "variant": variant,
                    "condition": cond_name, "layer": li, "pct_demographic": a,
                    "pct_instruction": b_, "pct_item_text": c_, "pct_other_scaffolding": e,
                })
            head_blocks.append(seg.astype(np.float16))
            head_index.append((item["item_id"], variant, cond_name))

    with torch.no_grad():
        for i, (_, item) in enumerate(candidates.iterrows()):
            prompt = build_prompt_variant(tok, "prefix", "baseline", item["text"])
            pred, prob, attn_lh, tokens = run_one(prompt)
            _, instr, txt = locate_spans(tokens, "prefix", "baseline")
            ok = _span_ok(instr) and txt is not None
            seg = segment_attention(attn_lh, None, instr, txt) if ok else None
            record(item, "all", "baseline", {"baseline"}, pred, prob, pred, prob, seg, ok)
            baselines[item["item_id"]] = (pred, prob)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(candidates)}", flush=True)

    bprob = pd.Series({k: v[1] for k, v in baselines.items()})
    in_band_ids = bprob[(bprob >= MIN_BASELINE_PROB) & (bprob <= MAX_BASELINE_PROB)].index
    items = candidates[candidates.item_id.isin(set(in_band_ids))].reset_index(drop=True)
    if MAX_ITEMS:
        items = items.head(MAX_ITEMS)
    print(f"\n{len(items)} of {len(candidates)} items have baseline P(yes) in "
          f"[{MIN_BASELINE_PROB}, {MAX_BASELINE_PROB}] -- these carry the condition grid.")
    print(f"  entropy in selected set: {items.entropy.min():.3f}-{items.entropy.max():.3f} "
          f"(NOT filtered on, recorded as a covariate)")
    print(f"  categories: {items['category'].value_counts().to_dict()}")
    items.to_csv(out_items, index=False)

    # top-N by entropy WITHIN the in-band set (not the full 500) -- has to be
    # recomputed per model, since a fixed item list wouldn't survive Qwen's much
    # stricter in-band filter (64 of 500 items here, vs 236 for Llama)
    illustrative = items.sort_values("entropy", ascending=False).head(N_ILLUSTRATIVE)
    print(f"\nIllustrative items for charts (top {N_ILLUSTRATIVE} by entropy, in-band): "
          f"{illustrative['item_id'].tolist()}")
    # item_id -> variant -> condition -> (tokens, weights, labels)
    chart_data = {}
    with torch.no_grad():
        for _, item in illustrative.iterrows():
            prompt = build_prompt_variant(tok, "prefix", "baseline", item["text"])
            _, _, attn_lh, tokens = run_one(prompt)
            _, instr, txt = locate_spans(tokens, "prefix", "baseline")
            if not (_span_ok(instr) and txt is not None):
                continue
            entry = content_chart_entry(attn_lh.mean(axis=(0, 1)), tokens, None, instr, txt)
            chart_data[item["item_id"]] = {v: {"baseline": entry} for v in CHART_VARIANTS}

    # --- stage 2: the condition grid ---------------------------------------
    grid_pairs = sorted(p for p in plan if p[1] != "baseline")
    n_total = len(items) * len(grid_pairs)
    print(f"\n=== Stage 2: {len(grid_pairs)} pairs x {len(items)} items = "
          f"{n_total} forward passes ===")
    n_done = 0
    t_start = time.time()
    with torch.no_grad():
        for _, item in items.iterrows():
            b_pred, b_prob = baselines[item["item_id"]]
            for (variant, cond) in grid_pairs:
                prompt = build_prompt_variant(tok, variant, cond, item["text"])
                pred, prob, attn_lh, tokens = run_one(prompt)
                demo, instr, txt = locate_spans(tokens, variant, cond)
                ok = demo is not None and _span_ok(instr) and txt is not None
                seg = segment_attention(attn_lh, demo, instr, txt) if ok else None
                record(item, variant, cond, plan[(variant, cond)], pred, prob,
                       b_pred, b_prob, seg, ok)
                if ok and item["item_id"] in chart_data and variant in CHART_VARIANTS and cond in CHART_CONDITIONS:
                    chart_data[item["item_id"]][variant][cond] = content_chart_entry(
                        attn_lh.mean(axis=(0, 1)), tokens, demo, instr, txt)
                n_done += 1
                if n_done % 500 == 0:
                    el = time.time() - t_start
                    rate = n_done / el
                    print(f"  {n_done}/{n_total}  elapsed {el/60:.1f} min  "
                          f"{rate:.2f} passes/s  ETA {(n_total - n_done)/rate/60:.1f} min", flush=True)
                if n_done % CHECKPOINT_EVERY == 0:
                    pd.DataFrame(result_rows).to_csv(out_main + ".partial", index=False)

    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(out_main, index=False)
    print(f"\nSaved {out_main}  ({len(results_df)} rows)")

    pd.DataFrame(layer_rows).to_csv(out_layer, index=False, compression="gzip")
    print(f"Saved {out_layer}  ({len(layer_rows)} rows)")

    head_arr = np.stack(head_blocks, axis=0)
    idx = pd.DataFrame(head_index, columns=["item_id", "variant", "condition"])
    np.savez_compressed(
        out_head, attention=head_arr, item_id=idx["item_id"].values,
        variant=idx["variant"].values.astype(str),
        condition=idx["condition"].values.astype(str),
        segments=np.array(SEGMENTS))
    print(f"Saved {out_head}  shape={head_arr.shape} (rows, layers, heads, segments), float16")

    bad = results_df[results_df["pct_other_scaffolding"].isna()]
    print(f"\nRows with unfound span: {len(bad)} (preflight should have made this 0)")
    if len(bad):
        print(bad[["item_id", "variant", "condition"]].drop_duplicates().to_string(index=False))

    grid = results_df[results_df.condition != "baseline"]
    print("\n=== Mean signed delta P(yes), prefix structure, all conditions ===")
    pref = grid[grid.variant == "prefix"].groupby(["axis", "condition"])["delta_prob_yes"]
    print(pref.agg(["mean", "count"]).round(4).sort_values("mean", ascending=False).to_string())

    print("\n=== Mean signed delta P(yes) by variant x condition ===")
    print(grid.groupby(["variant", "condition"])["delta_prob_yes"].mean().unstack().round(3).to_string())

    print("\n=== Mean sink-excluded demographic attention by variant x condition ===")
    print(grid.groupby(["variant", "condition"])["pct_demographic_norm"].mean().unstack().round(4).to_string())

    # --- illustrative content-attention charts: prefix vs infix vs suffix ---
    # Same phrase per condition across all three positions (verified locally to
    # tokenize to an identical length in each -- see conversation), so any
    # difference in the bars below is a position effect, not a length effect.
    charts_dir = f"{FIGURES_DIR}/content_attention_charts"
    os.makedirs(charts_dir, exist_ok=True)
    row_order = ["baseline"] + CHART_CONDITIONS
    n_saved = 0
    for item_id, per_variant in chart_data.items():
        for variant in CHART_VARIANTS:
            conds_present = [c for c in row_order if c in per_variant[variant]]
            if not conds_present:
                continue
            fig, axes = plt.subplots(len(conds_present), 1, figsize=(12, 3 * len(conds_present)), squeeze=False)
            for ax, cond_name in zip(axes[:, 0], conds_present):
                toks, weights, labels = per_variant[variant][cond_name]
                colors = [SEGMENT_COLORS[l] for l in labels]
                ax.bar(range(len(toks)), weights, color=colors)
                ax.set_xticks(range(len(toks)))
                ax.set_xticklabels([t.strip() or "·" for t in toks], rotation=60, ha="right", fontsize=8)
                ax.set_title(cond_name)
                ax.set_ylabel("attention\n(sink excluded, renorm.)")
            handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SEGMENT_COLORS.values()]
            fig.legend(handles, SEGMENT_COLORS.keys(), loc="upper right")
            fig.suptitle(f"model={MODEL_TAG}  item_id={item_id}  variant={variant}  "
                         f"(same phrase, position-only comparison)")
            fig.tight_layout()
            fig.savefig(f"{charts_dir}/item{item_id}_{variant}.png", dpi=150)
            plt.close(fig)
            n_saved += 1
    print(f"\nSaved {n_saved} illustrative content-attention charts "
          f"({len(chart_data)} items x {len(CHART_VARIANTS)} variants) to {charts_dir}/")
