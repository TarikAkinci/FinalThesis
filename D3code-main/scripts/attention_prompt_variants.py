"""
Prompt-sensitivity check, motivated by the placebo-controlled finding that "You are
X." prefix framing performs about as well as a content-free placebo for 3 of 4 tested
conditions (gender_man, gender_woman, region_western_europe), with region_arab_culture
as a partial exception. Before concluding the model is genuinely insensitive to
demographic content (rather than insensitive to THIS prompt's specific wording and
position), this reruns the same demographic-vs-placebo comparison under 4 prompt
structures:

  prefix   -- unchanged: "You are a woman. Text: \"...\"\nIs this text offensive?..."
  suffix   -- demographic clause moved to the END, after the instruction:
              "Text: \"...\"\nIs this text offensive? ... You are a woman."
  reworded -- same (prefix) position, different wording:
              "Assume you are a woman. Text: \"...\"\n..."
  embedded -- demographic folded directly into the question itself, no separate
              sentence: "Text: \"...\"\nAs a woman, is this text offensive? ..."

Uses the same ~100 high-disagreement items (from within the seeded 500-item D3CODE eval
sample) as attention_disagreement_scaled.py -- per your supervisor's instruction, both
studies now share one item pool instead of drawing separately. Same gender/region
conditions and placebo phrase as that study (religion excluded, it wasn't part of the
placebo comparison either). Baseline has no demographic clause, so it's identical across
all 4 variants -- computed once per item, not once per variant, to avoid redundant
forward passes.

Also fixes a real bug from the first version: the "reworded" phrases were hand-typed
independently and ended up length-asymmetric (every demographic phrase +1 word, the
placebo -2 words), which plausibly explained that variant's anomalous results on its
own. See _reword() below -- one mechanical transformation applied to all 5 phrases now.

Item selection now also requires baseline P(yes) in a reasonable band (see
select_items_with_baseline_filter.py, run first -- entropy-only selection turned out
to concentrate on items already pinned near-ceiling at baseline, leaving no room for
any perturbation to move the prediction up, only down).

Run as a batch job (needs a GPU) -- see attention_prompt_variants_job.sbatch.
"""
import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from transformers import AutoModelForCausalLM, AutoTokenizer

from attention_probe import find_token_span, find_span_by_delimiters
from zeroshot_two_datasets import PROMPT
from high_disagreement_items import load_filtered_items

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
MAX_LENGTH = 1024
RESULTS_DIR = "results/prompt_variants"
FIGURES_DIR = "figures/prompt_variants"
OUT_PATH = f"{RESULTS_DIR}/prompt_variants_results.csv"
# top-N-by-entropy items, same selection order load_filtered_items() returns -- matches
# the illustrative items attention_disagreement_scaled.py already charts (e.g. 899, 3263),
# so the two studies' figures show the same items throughout, not different ones by chance
N_ILLUSTRATIVE = 3
SEGMENT_COLORS = {"demographic": "tab:red", "instruction": "tab:blue", "item_text": "tab:green"}

INSTRUCTION_TEXT = "Is this text offensive? Answer with only 0 (no) or 1 (yes)."
# lowercase 'i' -- this is the same instruction text but mid-sentence in the embedded
# variant ("As a woman, is this text offensive?..."), so it needs its own literal match
INSTRUCTION_TEXT_EMBEDDED = "is this text offensive? Answer with only 0 (no) or 1 (yes)."
TEXT_PREFIX, _ = PROMPT.split("{text}")
# NOT the full original suffix ('"\nIs this text offensive?...') -- that assumes the
# instruction text immediately follows the item text, which is true for prefix/suffix/
# reworded but NOT for embedded ('"\nAs a woman, is this text offensive?...'). Every
# variant does still close the item text with a quote+newline before whatever comes
# next, so that's the one boundary that's actually constant across all 4 variants.
TEXT_SUFFIX = '"\n'

# override via e.g. VARIANTS=embedded to redo only one variant (its results overwrite
# only that variant's rows if you merge back with the previous CSV -- see conversation)
VARIANTS = os.environ.get("VARIANTS", "prefix,suffix,reworded,embedded").split(",")

def _reword(phrase):
    """Uniform +1-word transformation ('You are X.' -> 'Assume you are X.'),
    applied identically to every phrase including the placebo. The first
    version of this script hand-typed each reworded phrase independently and
    ended up asymmetric: every demographic phrase gained 1 word ("You are a
    woman." -> "Assume you are a woman.") but the placebo LOST 2 ("You are
    someone who prefers tea over coffee." -> "Assume you prefer tea over
    coffee."), which plausibly explained on its own why that variant looked
    more favorable to the demographic phrases than every other variant. A
    single mechanical transformation applied to all 5 phrases can't drift
    out of sync like that again."""
    assert phrase.startswith("You ")
    return "Assume you " + phrase[4:]


# condition -> (prefix/suffix phrase, noun phrase for the embedded variant)
BASE_CONDITIONS = {
    "gender_man": ("You are a man.", "a man"),
    "gender_woman": ("You are a woman.", "a woman"),
    "region_western_europe": ("You are someone from Western Europe.", "someone from Western Europe"),
    "region_arab_culture": ("You are someone from the Arab Culture region.", "someone from the Arab Culture region"),
    "placebo": ("You are someone who prefers tea over coffee.", "someone who prefers tea over coffee"),
}
# condition -> (prefix/suffix phrase, reworded phrase, noun phrase for the embedded variant)
TESTED_CONDITIONS = {
    cond: (phrase, _reword(phrase), noun) for cond, (phrase, noun) in BASE_CONDITIONS.items()
}


def build_content(variant, phrase_prefix, phrase_reworded, noun_phrase, text):
    base = PROMPT.format(text=text)
    if variant == "prefix":
        return f"{phrase_prefix} {base}" if phrase_prefix else base
    if variant == "suffix":
        return f"{base} {phrase_prefix}" if phrase_prefix else base
    if variant == "reworded":
        return f"{phrase_reworded} {base}" if phrase_reworded else base
    if variant == "embedded":
        if not noun_phrase:
            return base
        return f'Text: "{text}"\nAs {noun_phrase}, {INSTRUCTION_TEXT_EMBEDDED}'
    raise ValueError(variant)


def build_prompt_variant(tok, variant, phrase_prefix, phrase_reworded, noun_phrase, text):
    content = build_content(variant, phrase_prefix, phrase_reworded, noun_phrase, text)
    messages = [{"role": "user", "content": content}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # entropy-ranked AND baseline-probability-band-filtered, same item set
    # attention_disagreement_scaled.py uses -- run
    # select_items_with_baseline_filter.py first
    df = load_filtered_items()
    print(f"Selected {len(df)} high-disagreement items (entropy range "
          f"{df.entropy.min():.3f}-{df.entropy.max():.3f}): {df['category'].value_counts().to_dict()}")

    print(f"\nLoading model: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    )
    model.eval()

    zero_id = tok.encode("0", add_special_tokens=False)[0]
    one_id = tok.encode("1", add_special_tokens=False)[0]

    def run_one(prompt_text):
        inputs = tok(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH,
                     add_special_tokens=False).to(model.device)
        out = model(**inputs, output_attentions=True)
        # No .generate()/temperature involved: a single deterministic forward pass,
        # equivalent to temperature=0/greedy decoding, no sampling.
        logits = out.logits[0, -1]
        predicted = "1" if logits[one_id] > logits[zero_id] else "0"
        _, prob_one = torch.softmax(logits[[zero_id, one_id]].float(), dim=0)
        attn_stack = torch.stack(out.attentions, dim=0)
        attn = attn_stack[:, 0, :, -1, :].float().mean(dim=(0, 1)).cpu().numpy()
        input_ids = inputs["input_ids"][0].tolist()
        tokens = [tok.decode([tid]) for tid in input_ids]
        return predicted, prob_one.item(), attn, tokens

    def content_chart_entry(attn, tokens, demo_span, instr_span, text_span):
        """Sink-excluded, renormalized attention over just the demographic/instruction/
        item-text tokens -- same construction as the other scripts' content-only bar
        charts. Returns (tokens, weights, labels) for one condition's bar chart row."""
        spans = [("demographic", demo_span), ("instruction", instr_span), ("item_text", text_span)]
        content_idx = sorted({i for _, span in spans if span for i in range(span[0], span[1] + 1)})
        idx_to_label = {i: label for label, span in spans if span for i in range(span[0], span[1] + 1)}
        content_attn = np.array([attn[i] for i in content_idx])
        content_attn_norm = content_attn / content_attn.sum() if content_attn.sum() > 0 else content_attn
        return [tokens[i] for i in content_idx], content_attn_norm, [idx_to_label[i] for i in content_idx]

    result_rows = []
    n_total = len(df) * (1 + len(VARIANTS) * len(TESTED_CONDITIONS))
    n_done = 0

    illustrative_items = set(df.head(N_ILLUSTRATIVE)["item_id"])  # top-entropy items, already sorted
    # item_id -> variant -> condition -> (tokens, weights, labels); baseline is identical
    # across variants (same prompt), so its entry gets copied into every variant's chart
    chart_data = {}

    with torch.no_grad():
        for _, row in df.iterrows():
            text = row["text"]
            baseline_prompt = build_prompt_variant(tok, "prefix", None, None, None, text)
            b_pred, b_prob, b_attn, b_tokens = run_one(baseline_prompt)
            n_done += 1

            if row["item_id"] in illustrative_items:
                b_instr_span = find_token_span(b_tokens, INSTRUCTION_TEXT)
                b_text_span = find_span_by_delimiters(b_tokens, TEXT_PREFIX, TEXT_SUFFIX)
                if b_instr_span is not None and b_text_span is not None:
                    baseline_entry = content_chart_entry(b_attn, b_tokens, None, b_instr_span, b_text_span)
                    chart_data[row["item_id"]] = {v: {"baseline": baseline_entry} for v in VARIANTS}

            for variant in VARIANTS:
                for cond_name, (phrase_prefix, phrase_reworded, noun_phrase) in TESTED_CONDITIONS.items():
                    prompt = build_prompt_variant(tok, variant, phrase_prefix, phrase_reworded, noun_phrase, text)
                    predicted, prob_yes, attn, tokens = run_one(prompt)

                    if variant == "embedded":
                        demo_span = find_token_span(tokens, noun_phrase)
                        instr_span = find_token_span(tokens, INSTRUCTION_TEXT_EMBEDDED)
                    else:
                        search_phrase = phrase_reworded if variant == "reworded" else phrase_prefix
                        demo_span = find_token_span(tokens, search_phrase)
                        instr_span = find_token_span(tokens, INSTRUCTION_TEXT)
                    text_span = find_span_by_delimiters(tokens, TEXT_PREFIX, TEXT_SUFFIX)

                    if demo_span is None or instr_span is None or text_span is None:
                        pct_demo = pct_instr = pct_text = pct_other = np.nan
                    else:
                        pct_demo = float(attn[demo_span[0]: demo_span[1] + 1].sum())
                        pct_instr = float(attn[instr_span[0]: instr_span[1] + 1].sum())
                        pct_text = float(attn[text_span[0]: text_span[1] + 1].sum())
                        pct_other = 1.0 - pct_demo - pct_instr - pct_text

                    content_mass = 1.0 - pct_other if not np.isnan(pct_other) else np.nan
                    pct_demo_norm = pct_demo / content_mass if content_mass and content_mass > 0 else np.nan

                    if (row["item_id"] in chart_data and demo_span is not None
                            and instr_span is not None and text_span is not None):
                        chart_data[row["item_id"]][variant][cond_name] = content_chart_entry(
                            attn, tokens, demo_span, instr_span, text_span)

                    result_rows.append({
                        "item_id": row["item_id"], "category": row["category"], "variant": variant,
                        "condition": cond_name, "predicted": predicted, "prob_yes": prob_yes,
                        "baseline_pred": b_pred, "baseline_prob_yes": b_prob,
                        "flipped_vs_baseline": predicted != b_pred, "delta_prob_yes": prob_yes - b_prob,
                        "pct_demographic": pct_demo, "pct_instruction": pct_instr, "pct_item_text": pct_text,
                        "pct_other_scaffolding": pct_other, "pct_demographic_norm": pct_demo_norm,
                    })
                    n_done += 1
                    if n_done % 50 == 0:
                        print(f"  {n_done}/{n_total}")

    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH}")

    bad = results_df[results_df["pct_demographic_norm"].isna()]
    print(f"\nRows with unfound span: {len(bad)}")
    if len(bad):
        print(bad[["item_id", "variant", "condition"]].to_string(index=False))

    print("\n=== Mean pct_demographic_norm by variant x condition ===")
    print(results_df.groupby(["variant", "condition"])["pct_demographic_norm"].mean().unstack())

    print("\n=== Demographic vs placebo, per variant (paired Wilcoxon, pct_demographic_norm) ===")
    for variant in VARIANTS:
        vdf = results_df[results_df.variant == variant]
        placebo = vdf[vdf.condition == "placebo"].set_index("item_id")["pct_demographic_norm"]
        for cond in ["gender_man", "gender_woman", "region_western_europe", "region_arab_culture"]:
            demo = vdf[vdf.condition == cond].set_index("item_id")["pct_demographic_norm"]
            merged = pd.DataFrame({"demo": demo, "placebo": placebo}).dropna()
            if len(merged) < 5 or (merged["demo"] == merged["placebo"]).all():
                print(f"  {variant:10s} {cond:24s}: insufficient/degenerate data (n={len(merged)})")
                continue
            diff = merged["demo"] - merged["placebo"]
            stat, p = wilcoxon(diff)
            print(f"  {variant:10s} {cond:24s}: mean_diff={diff.mean():+.4f}  "
                  f"frac_higher={(diff > 0).mean():.2f}  n={len(diff)}  p={p:.4g}")

    print("\n=== Demographic vs placebo, per variant (paired Wilcoxon, |delta_prob_yes|) ===")
    for variant in VARIANTS:
        vdf = results_df[results_df.variant == variant]
        placebo = vdf[vdf.condition == "placebo"].set_index("item_id")["delta_prob_yes"].abs()
        for cond in ["gender_man", "gender_woman", "region_western_europe", "region_arab_culture"]:
            demo = vdf[vdf.condition == cond].set_index("item_id")["delta_prob_yes"].abs()
            merged = pd.DataFrame({"demo": demo, "placebo": placebo}).dropna()
            if len(merged) < 5 or (merged["demo"] == merged["placebo"]).all():
                print(f"  {variant:10s} {cond:24s}: insufficient/degenerate data (n={len(merged)})")
                continue
            diff = merged["demo"] - merged["placebo"]
            stat, p = wilcoxon(diff)
            print(f"  {variant:10s} {cond:24s}: mean_diff={diff.mean():+.4f}  "
                  f"frac_higher={(diff > 0).mean():.2f}  n={len(diff)}  p={p:.4g}")

    # illustrative content-attention bar charts: same top-entropy items already charted in
    # attention_disagreement_scaled.py (e.g. 899, 3263), one chart per item PER VARIANT so
    # you can compare how the same item's attention breakdown looks under prefix/suffix/
    # reworded/embedded side by side, not just under the plain prefix structure.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts_dir = f"{FIGURES_DIR}/content_attention_charts"
    os.makedirs(charts_dir, exist_ok=True)
    row_order = ["baseline"] + list(TESTED_CONDITIONS.keys())  # baseline, gender_man, gender_woman, region_*, placebo
    n_saved = 0
    for item_id, per_variant in chart_data.items():
        for variant in VARIANTS:
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
            fig.suptitle(f"item_id={item_id}  variant={variant}")
            fig.tight_layout()
            fig.savefig(f"{charts_dir}/item{item_id}_{variant}.png", dpi=150)
            plt.close(fig)
            n_saved += 1
    print(f"\nSaved {n_saved} illustrative content-attention charts (items x variants) to {charts_dir}/")
