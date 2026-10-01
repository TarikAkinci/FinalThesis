"""
Attention probing per supervisor's actual plan (steering excluded):
1. Where does Llama pay attention on the plain classification prompt?
2. Does attention shift toward demographic-descriptor tokens when added to
   the prompt, and how does attention over the item text itself change?

Uses Inseq's "attention" attribution method (verified locally against a
small model first -- see conversation). For a decoder-only model, the actual
data lives in `target_attributions`, shape
[seq_len, n_generated_tokens, n_layers, n_heads] -- NOT `source_attributions`,
which is None here (that field is for encoder-decoder models).

Demographic conditions are focused contrast pairs, grounded in D3CODE's own
rater categories where they exist (Gender, Region), plus religion (not a
D3CODE field, so Christian/Muslim chosen independently -- worth flagging to
your supervisor since it's not dataset-grounded like the other two).

Items: stratified sample across D3CODE's 3 categories (moral/random/
social-group), drawn from the frozen test split, matching project convention.

Run as a batch job (see attention_probe_job.sbatch) -- needs a GPU.
"""
import json
import os
import numpy as np
import pandas as pd
import inseq
from transformers import AutoTokenizer

from zeroshot_two_datasets import PROMPT
from span_utils import find_token_span, find_span_by_delimiters  # noqa: F401 -- re-exported, see span_utils.py

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
D3_ITEMS_CSV = "../dataset/d3-items.csv"
D3_SPLITS_JSON = "../splits.json"
N_PER_CATEGORY = 10
SEED = 42
OUT_PATH = "attention_probe_results.csv"

CONDITIONS = {
    "baseline": None,
    "gender_man": "You are a man.",
    "gender_woman": "You are a woman.",
    "region_western_europe": "You are someone from Western Europe.",
    "region_arab_culture": "You are someone from the Arab Culture region.",
    "religion_christian": "You are Christian.",
    "religion_muslim": "You are Muslim.",
}


def prepare_items(n_per_category, seed):
    items = pd.read_csv(D3_ITEMS_CSV)
    with open(D3_SPLITS_JSON) as f:
        test_ids = set(json.load(f)["test"])
    df = items[items["item_id"].isin(test_ids)].copy()
    sampled = df.groupby("category", group_keys=False).apply(
        lambda g: g.sample(n=min(n_per_category, len(g)), random_state=seed)
    )
    return sampled.reset_index(drop=True)


def build_prompt(tok, condition_prefix, text):
    content = PROMPT.format(text=text)
    if condition_prefix:
        content = f"{condition_prefix} {content}"
    messages = [{"role": "user", "content": content}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


if __name__ == "__main__":
    df = prepare_items(N_PER_CATEGORY, SEED)
    print(f"Sampled {len(df)} items across categories: {df['category'].value_counts().to_dict()}")

    print(f"\nLoading tokenizer + Inseq model: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    inseq_model = inseq.load_model(MODEL_NAME, "attention")

    # Cluster's Python 3.9 resolves to inseq==0.6.0 (latest, 0.7.1, needs a
    # newer Python than the cluster has). 0.6.0 is missing a safeguard that
    # 0.7.1 already has: HuggingfaceModel.__init__ reads
    # model.config.eos_token_id and stores it as self.eos_token_id without
    # checking whether it's a list. Llama-3.1's config defines THREE eos
    # tokens (a documented multi-eos setup), so self.eos_token_id ends up as
    # a 3-int list, which crashes inseq's internal baseline computation
    # (`batch["input_ids"].ne(self.eos_token_id)` -- .ne() needs a single
    # value, not a list). Verified by diffing the actual installed 0.6.0
    # source against 0.7.1's: 0.7.1 adds exactly this isinstance check. This
    # reproduces that same fix as a two-line patch post-load, since by the
    # time load_model() returns, the bad value is already cached on the
    # instance -- patching model.config afterward wouldn't help.
    if isinstance(inseq_model.eos_token_id, list):
        print(f"patching inseq eos_token_id (was {inseq_model.eos_token_id}, a known inseq==0.6.0 bug on multi-eos models)")
        inseq_model.eos_token_id = inseq_model.eos_token_id[0]

    # Same encode() line, second unguarded assumption: it also does
    # `... * self.tokenizer.unk_token_id`, uncritically assuming a UNK token
    # exists. Llama 3's tokenizer (byte-level BPE, like GPT-2's) has none --
    # any input is representable without a fallback token, so unk_token_id is
    # None by design, not a bug in the tokenizer. This baseline computation
    # isn't even used by the "attention" method (only gradient-based methods
    # need a substitution baseline), so the exact fallback value doesn't
    # matter for our results -- it just needs to not be None so encode()
    # can finish. Verified locally that plain reassignment works (it's a
    # regular attribute, not a read-only property).
    if inseq_model.tokenizer.unk_token_id is None:
        fallback = inseq_model.tokenizer.pad_token_id or inseq_model.eos_token_id or 0
        print(f"patching inseq tokenizer unk_token_id (was None, no UNK token in this tokenizer) -> {fallback}")
        inseq_model.tokenizer.unk_token_id = fallback

    zero_id_str, one_id_str = "0", "1"
    results = []
    n_total = len(df) * len(CONDITIONS)
    n_done = 0

    for _, row in df.iterrows():
        for cond_name, cond_prefix in CONDITIONS.items():
            prompt = build_prompt(tok, cond_prefix, row["text"])
            # skip_special_tokens=False: attribute()'s internal generate() call
            # decodes with skip_special_tokens=True by default, stripping
            # Llama's literal chat-template tokens (<|begin_of_text|>,
            # <|start_header_id|>...) from the decoded text. Our input_texts
            # string (from apply_chat_template) still contains those tokens
            # as literal text, so attribute()'s internal assertion that
            # generated_texts.startswith(input_texts) fails on a pure text
            # mismatch -- unrelated to attention content, just a decode-vs-
            # source formatting mismatch. Forcing this off keeps both sides
            # on the same footing. Diagnosed by reading the actual 0.6.0
            # source (attribution_model.py:432-435, huggingface_model.py's
            # generate()); not independently reproduced locally since the
            # small model used for local testing doesn't register its own
            # chat tokens as "special" the same way Llama's tokenizer does.
            out = inseq_model.attribute(
                input_texts=prompt,
                generation_args={"max_new_tokens": 1, "skip_special_tokens": False},
                show_progress=False,
            )
            attr = out.sequence_attributions[0]
            tokens = [t.token for t in attr.target]
            predicted = tokens[-1].strip()

            # attn: [seq_len, n_generated=1, n_layers, n_heads] -> mean over layers/heads
            attn = attr.target_attributions[:, 0, :, :].float().mean(axis=(1, 2)).numpy()
            attn = attn[: len(tokens) - 1]  # drop the row for the generated token itself

            pct_on_demographic = np.nan
            if cond_prefix:
                span = find_token_span(tokens, cond_prefix)
                if span:
                    pct_on_demographic = float(attn[span[0]: span[1] + 1].sum())

            text_span = find_token_span(tokens, row["text"])
            pct_on_item_text = float(attn[text_span[0]: text_span[1] + 1].sum()) if text_span else np.nan

            results.append({
                "item_id": row["item_id"], "category": row["category"], "condition": cond_name,
                "predicted": predicted, "pct_attention_on_demographic": pct_on_demographic,
                "pct_attention_on_item_text": pct_on_item_text,
            })
            n_done += 1
            if n_done % 20 == 0:
                print(f"  {n_done}/{n_total}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH}")
    print("\nMean %% attention on item text, by condition:")
    print(results_df.groupby("condition")["pct_attention_on_item_text"].mean().sort_values(ascending=False))
    print("\nMean %% attention on demographic phrase, by condition (baseline excluded):")
    print(results_df.dropna(subset=["pct_attention_on_demographic"]).groupby("condition")["pct_attention_on_demographic"].mean())
