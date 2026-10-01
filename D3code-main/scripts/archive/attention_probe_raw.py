"""
Fallback for attention_probe.py: Inseq's "attention" method crashes on
Llama-3.1-8B-Instruct with a TypeError inside its own encode() (a list of 3
eos_token_ids reaching a .ne() call that expects a single int -- a library
compatibility bug, not something in our code). Per the earlier plan ("try
Inseq first, fall back to raw output_attentions=True if it doesn't work"),
this reimplements the same analysis directly against the model's own
output_attentions, which we already verified works on Llama in an earlier
exploration.

Same items, same demographic CONDITIONS, same token-span-finding logic as
attention_probe.py -- reused via import, not duplicated. Only the attention
SOURCE differs: a single forward pass with output_attentions=True instead of
Inseq's attribution wrapper.

Run as a batch job (see attention_probe_raw_job.sbatch) -- needs a GPU.
"""
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attention_probe import CONDITIONS, prepare_items, find_token_span, find_span_by_delimiters, build_prompt, N_PER_CATEGORY, SEED
from zeroshot_two_datasets import PROMPT

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
OUT_PATH = "attention_probe_results.csv"
MAX_LENGTH = 1024  # was 512 -- headroom against truncation for any unusually long item
# Item text located by its fixed surrounding delimiters, not its own content --
# see attention_probe_detailed.py for why (D3CODE item text can contain quote
# characters that collide with PROMPT's own wrapping quotes).
TEXT_PREFIX, TEXT_SUFFIX = PROMPT.split("{text}")


if __name__ == "__main__":
    df = prepare_items(N_PER_CATEGORY, SEED)
    print(f"Sampled {len(df)} items across categories: {df['category'].value_counts().to_dict()}")

    print(f"\nLoading model: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    # eager attention is required for output_attentions=True to work at all --
    # the default SDPA implementation doesn't expose attention weights (verified
    # locally: it silently returns an empty attentions tuple, not an error)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    )
    model.eval()

    zero_id = tok.encode("0", add_special_tokens=False)[0]
    one_id = tok.encode("1", add_special_tokens=False)[0]

    results = []
    n_total = len(df) * len(CONDITIONS)
    n_done = 0

    with torch.no_grad():
        for _, row in df.iterrows():
            for cond_name, cond_prefix in CONDITIONS.items():
                prompt = build_prompt(tok, cond_prefix, row["text"])
                # add_special_tokens=False: apply_chat_template(tokenize=False) already
                # writes <|begin_of_text|> as literal text into `prompt`. Without this,
                # tok()'s default add_special_tokens=True prepends a SECOND BOS token.
                inputs = tok(
                    prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH,
                    add_special_tokens=False,
                ).to(model.device)
                out = model(**inputs, output_attentions=True)

                # No .generate()/temperature involved: a single deterministic forward
                # pass, predicted label via direct logit comparison -- equivalent to
                # temperature=0/greedy decoding, no sampling.
                logits = out.logits[0, -1]
                predicted = "1" if logits[one_id] > logits[zero_id] else "0"

                # out.attentions: tuple of n_layers tensors, each [1, n_heads, seq, seq].
                # Row [-1, :] is the last-token position's attention over the whole
                # prompt -- exactly the distribution used to produce the next token.
                attn_stack = torch.stack(out.attentions, dim=0)          # [n_layers, 1, n_heads, seq, seq]
                attn = attn_stack[:, 0, :, -1, :].float().mean(dim=(0, 1)).cpu().numpy()  # [seq] -- mean over layers/heads

                input_ids = inputs["input_ids"][0].tolist()
                tokens = [tok.decode([tid]) for tid in input_ids]

                pct_on_demographic = np.nan
                if cond_prefix:
                    span = find_token_span(tokens, cond_prefix)
                    if span:
                        pct_on_demographic = float(attn[span[0]: span[1] + 1].sum())

                text_span = find_span_by_delimiters(tokens, TEXT_PREFIX, TEXT_SUFFIX)
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
