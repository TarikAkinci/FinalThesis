"""
Extends attention_probe_raw.py to actually answer the two questions your
supervisor asked first: WHERE does the model pay attention in a simple
prompt, and HOW does that change when demographic info is added.

Previous version only kept two aggregate percentages (item text, demographic
phrase) and threw away the full per-token attention array. This version:
1. Saves the FULL per-token (token, attention_weight) breakdown for every
   item/condition -- costs nothing extra, since we're already running the
   same 210 forward passes; we were just discarding most of the data before.
2. Adds a segment-level breakdown: demographic phrase / fixed instruction
   text / item text / everything else (chat-template scaffolding), instead
   of only item-text-vs-demographic. The instruction text is a fixed,
   always-present substring (INSTRUCTION_TEXT below), located the same way
   as the other spans.
3. Renders an actual attention heatmap (bertviz's head_view) for a handful of
   example items, reusing the SAME forward pass already computed for the
   segment/token analysis -- no extra GPU cost, just saving what we already
   have in a different form. bertviz only visualizes tensors we already
   trust (from output_attentions) -- it never touches model loading,
   tokenization, or generation, so it carries none of the risk Inseq did.

Run as a batch job (see attention_probe_detailed_job.sbatch) -- needs a GPU.
Needs: pip install bertviz   (add to thesis-env on the cluster)
"""
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bertviz import head_view

from attention_probe import CONDITIONS, prepare_items, find_token_span, find_span_by_delimiters, build_prompt, N_PER_CATEGORY, SEED
from zeroshot_two_datasets import PROMPT

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
RESULTS_DIR = "results/attention_probing"
FIGURES_DIR = "figures/attention_probing"
SEGMENTS_OUT_PATH = f"{RESULTS_DIR}/attention_probe_segments.csv"
TOKENS_OUT_PATH = f"{RESULTS_DIR}/attention_probe_token_level.csv"
LAYERWISE_OUT_PATH = f"{RESULTS_DIR}/attention_probe_layerwise.csv"
HEATMAP_CONDITIONS = ["baseline", "religion_muslim", "religion_christian"]  # the length-matched contrast with the largest effect
MAX_LENGTH = 1024  # was 512 -- headroom against truncation for any unusually long item

# The fixed part of PROMPT, independent of item text -- locate this span the
# same way as the demographic phrase.
INSTRUCTION_TEXT = "Is this text offensive? Answer with only 0 (no) or 1 (yes)."
# Item text is located by its FIXED surrounding delimiters, not its own content
# (find_span_by_delimiters), derived straight from PROMPT so they can't drift
# out of sync with it. Plain substring search on the item's own text
# (find_token_span) is NOT safe here: D3CODE item text is arbitrary and can
# itself contain quote characters that collide with PROMPT's own wrapping
# quotes -- e.g. item_id=1560's text starts and ends with a literal '"',
# which broke span-finding for that item (see conversation: it showed up as a
# missing item-text segment in that item's content-attention bar chart).
TEXT_PREFIX, TEXT_SUFFIX = PROMPT.split("{text}")


if __name__ == "__main__":
    df = prepare_items(N_PER_CATEGORY, SEED)
    print(f"Sampled {len(df)} items across categories: {df['category'].value_counts().to_dict()}")

    print(f"\nLoading model: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    )
    model.eval()

    zero_id = tok.encode("0", add_special_tokens=False)[0]
    one_id = tok.encode("1", add_special_tokens=False)[0]

    example_items = set(df.groupby("category").head(2)["item_id"].tolist())

    segment_rows = []
    token_rows = []
    layer_rows = []
    heatmap_data = {}  # (item_id, condition) -> (attentions tuple, tokens) for bertviz, example items only
    chart_data = {}  # (item_id, condition) -> (content_tokens, content_attn_norm, segment_labels), for the content-only bar chart
    n_total = len(df) * len(CONDITIONS)
    n_done = 0

    with torch.no_grad():
        for _, row in df.iterrows():
            for cond_name, cond_prefix in CONDITIONS.items():
                prompt = build_prompt(tok, cond_prefix, row["text"])
                # add_special_tokens=False: apply_chat_template(tokenize=False) already
                # writes <|begin_of_text|> as literal text into `prompt`. Without this,
                # tok()'s default add_special_tokens=True prepends a SECOND BOS token,
                # which is what produced the duplicate <|begin_of_text|> seen in the
                # bertviz heatmaps.
                inputs = tok(
                    prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH,
                    add_special_tokens=False,
                ).to(model.device)
                out = model(**inputs, output_attentions=True)

                # No .generate()/temperature involved anywhere in this script: a single
                # deterministic forward pass, predicted label from a direct logit
                # comparison (argmax over the two candidate tokens). Equivalent to
                # temperature=0/greedy decoding -- same input always gives the same
                # attention weights and the same predicted label, no sampling.
                logits = out.logits[0, -1]
                predicted = "1" if logits[one_id] > logits[zero_id] else "0"

                attn_stack = torch.stack(out.attentions, dim=0)
                attn = attn_stack[:, 0, :, -1, :].float().mean(dim=(0, 1)).cpu().numpy()
                # same data, head-averaged but NOT layer-averaged -- lets us check whether
                # the sink and the demographic effect are uniform across layers or
                # concentrated in specific ones (every paper we summarized found the latter)
                attn_per_layer = attn_stack[:, 0, :, -1, :].float().mean(dim=1).cpu().numpy()  # [n_layers, seq]

                input_ids = inputs["input_ids"][0].tolist()
                tokens = [tok.decode([tid]) for tid in input_ids]

                if row["item_id"] in example_items and cond_name in HEATMAP_CONDITIONS:
                    # keep the full per-layer/per-head attentions (moved to CPU) for bertviz later --
                    # cheap for this small subset, reuses the same forward pass, no extra GPU cost
                    heatmap_data[(row["item_id"], cond_name)] = (
                        tuple(a.cpu() for a in out.attentions), tokens,
                    )

                # save the full per-token breakdown -- no extra forward pass, just keeping what we already computed
                for pos, (t, a) in enumerate(zip(tokens, attn)):
                    token_rows.append({
                        "item_id": row["item_id"], "condition": cond_name,
                        "position": pos, "token": t, "attention": float(a),
                    })

                demo_span = find_token_span(tokens, cond_prefix) if cond_prefix else None
                instr_span = find_token_span(tokens, INSTRUCTION_TEXT)
                text_span = find_span_by_delimiters(tokens, TEXT_PREFIX, TEXT_SUFFIX)
                if instr_span is None or text_span is None:
                    print(f"  WARNING: span not found for item_id={row['item_id']} condition={cond_name} "
                          f"(instr_span={instr_span}, text_span={text_span}, prompt_tokens={len(tokens)}) -- "
                          f"likely truncated by MAX_LENGTH or an unexpected text encoding; segment/layer "
                          f"percentages for this row will be NaN")

                pct_demo = float(attn[demo_span[0]: demo_span[1] + 1].sum()) if demo_span else np.nan
                pct_instr = float(attn[instr_span[0]: instr_span[1] + 1].sum()) if instr_span else np.nan
                pct_text = float(attn[text_span[0]: text_span[1] + 1].sum()) if text_span else np.nan
                pct_other = 1.0 - pct_text - pct_instr - (pct_demo if not np.isnan(pct_demo) else 0.0)

                # Per-layer version of the same 4-way segment breakdown -- identifies
                # whether the sink and any demographic-attention effect are spread
                # uniformly across all 32 layers or concentrated in a subset (needed to
                # target specific layers/heads if/when we move to the steering phase).
                for l in range(attn_per_layer.shape[0]):
                    layer_attn = attn_per_layer[l]
                    l_demo = float(layer_attn[demo_span[0]: demo_span[1] + 1].sum()) if demo_span else np.nan
                    l_instr = float(layer_attn[instr_span[0]: instr_span[1] + 1].sum()) if instr_span else np.nan
                    l_text = float(layer_attn[text_span[0]: text_span[1] + 1].sum()) if text_span else np.nan
                    l_other = (1.0 - l_text - l_instr - (l_demo if not np.isnan(l_demo) else 0.0)
                               if not (np.isnan(l_text) or np.isnan(l_instr)) else np.nan)
                    layer_rows.append({
                        "item_id": row["item_id"], "condition": cond_name, "layer": l,
                        "pct_demographic": l_demo, "pct_instruction": l_instr,
                        "pct_item_text": l_text, "pct_other_scaffolding": l_other,
                    })

                # Sink-excluded, renormalized shares: pct_other_scaffolding is dominated
                # by the attention-sink effect (Xiao et al., 2023) parking most mass on
                # the fixed chat-template tokens (BOS, header tokens, etc.), which makes
                # the raw demo/instr/text percentages hard to compare across conditions --
                # a small absolute shift can be a large relative one once the sink mass is
                # factored out. Renormalize over only the non-scaffolding attention.
                content_mass = 1.0 - pct_other
                pct_demo_norm = (pct_demo / content_mass) if (content_mass > 0 and not np.isnan(pct_demo)) else np.nan
                pct_instr_norm = pct_instr / content_mass if content_mass > 0 else np.nan
                pct_text_norm = pct_text / content_mass if content_mass > 0 else np.nan

                if row["item_id"] in example_items and cond_name in HEATMAP_CONDITIONS:
                    # Same sink-exclusion as above, but kept at the per-token level (not
                    # just span sums) so we can plot a readable, sink-free bar chart --
                    # bertviz's head_view can't show this contrast, it's crowded out by
                    # the ~80% sink connection at every layer.
                    spans = [("demographic", demo_span), ("instruction", instr_span), ("item_text", text_span)]
                    content_idx = sorted({i for _, span in spans if span for i in range(span[0], span[1] + 1)})
                    idx_to_label = {}
                    for label, span in spans:
                        if span:
                            for i in range(span[0], span[1] + 1):
                                idx_to_label[i] = label
                    content_attn = np.array([attn[i] for i in content_idx])
                    content_attn_norm = content_attn / content_attn.sum() if content_attn.sum() > 0 else content_attn
                    chart_data[(row["item_id"], cond_name)] = (
                        [tokens[i] for i in content_idx],
                        content_attn_norm,
                        [idx_to_label[i] for i in content_idx],
                    )

                segment_rows.append({
                    "item_id": row["item_id"], "category": row["category"], "condition": cond_name,
                    "predicted": predicted, "pct_demographic": pct_demo, "pct_instruction": pct_instr,
                    "pct_item_text": pct_text, "pct_other_scaffolding": pct_other,
                    "pct_demographic_norm": pct_demo_norm, "pct_instruction_norm": pct_instr_norm,
                    "pct_item_text_norm": pct_text_norm,
                })
                n_done += 1
                if n_done % 20 == 0:
                    print(f"  {n_done}/{n_total}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    segments_df = pd.DataFrame(segment_rows)
    segments_df.to_csv(SEGMENTS_OUT_PATH, index=False)
    pd.DataFrame(token_rows).to_csv(TOKENS_OUT_PATH, index=False)
    layerwise_df = pd.DataFrame(layer_rows)
    layerwise_df.to_csv(LAYERWISE_OUT_PATH, index=False)
    print(f"\nSaved {SEGMENTS_OUT_PATH}, {TOKENS_OUT_PATH}, and {LAYERWISE_OUT_PATH}")

    print("\nMean attention by segment, by condition (this is the 'where does attention go' answer):")
    print(segments_df.groupby("condition")[["pct_demographic", "pct_instruction", "pct_item_text", "pct_other_scaffolding"]].mean())

    print("\nSame breakdown, sink-excluded and renormalized over only non-scaffolding attention")
    print("(factors out the attention-sink effect -- fairer cross-condition comparison):")
    print(segments_df.groupby("condition")[["pct_demographic_norm", "pct_instruction_norm", "pct_item_text_norm"]].mean())

    print("\nSink (pct_other_scaffolding) by layer, baseline condition (is the sink uniform across layers?):")
    print(layerwise_df[layerwise_df.condition == "baseline"].groupby("layer")["pct_other_scaffolding"].mean())

    print("\nDemographic attention by layer, religion_muslim vs religion_christian")
    print("(where does the demographic-attention effect concentrate?):")
    layer_demo = layerwise_df[layerwise_df.condition.isin(["religion_muslim", "religion_christian"])]
    print(layer_demo.groupby(["layer", "condition"])["pct_demographic"].mean().unstack())

    # Concrete, quotable examples for the report: top-10 attended words for a
    # few items, baseline vs. the two conditions with the clearest
    # length-matched contrast (religion showed the largest per-token effect
    # in the earlier normalized analysis).
    tokens_df = pd.DataFrame(token_rows)

    print("\n--- Top-10 attended tokens, example items, baseline vs. religion_muslim vs. religion_christian ---")
    for item_id in sorted(example_items):
        item_text = df[df["item_id"] == item_id]["text"].values[0]
        print(f"\nitem_id={item_id}  text={item_text[:80]!r}")
        for cond in ["baseline", "religion_muslim", "religion_christian"]:
            sub = tokens_df[(tokens_df["item_id"] == item_id) & (tokens_df["condition"] == cond)]
            top10 = sub.sort_values("attention", ascending=False).head(10)
            words = [f"{r.token.strip()!r}:{r.attention:.3f}" for r in top10.itertuples()]
            print(f"  [{cond:20s}] {', '.join(words)}")

    # actual attention heatmaps, one HTML file per (item, condition) captured above --
    # open these in a browser; they're the visual, report-ready version of the printed lists
    heatmaps_dir = f"{FIGURES_DIR}/attention_heatmaps"
    os.makedirs(heatmaps_dir, exist_ok=True)
    for (item_id, cond_name), (attentions, tokens) in heatmap_data.items():
        html = head_view(attentions, tokens, html_action="return")
        path = f"{heatmaps_dir}/item{item_id}_{cond_name}.html"
        with open(path, "w") as f:
            f.write(html.data)
    print(f"\nSaved {len(heatmap_data)} attention heatmaps to {heatmaps_dir}/")

    # Content-only bar charts: bertviz's head_view can't show the demographic-attention
    # shift because it's crowded out by the ~80% attention-sink connection at every
    # layer (see conversation). These plot the SAME per-item data restricted to just
    # the demographic/instruction/item-text tokens, sink excluded and renormalized --
    # the visual counterpart of the pct_*_norm numbers above, readable at a glance.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SEGMENT_COLORS = {"demographic": "tab:red", "instruction": "tab:blue", "item_text": "tab:green"}
    charts_dir = f"{FIGURES_DIR}/content_attention_charts"
    os.makedirs(charts_dir, exist_ok=True)
    for item_id in sorted(example_items):
        conds_present = [c for c in HEATMAP_CONDITIONS if (item_id, c) in chart_data]
        if not conds_present:
            continue
        fig, axes = plt.subplots(len(conds_present), 1, figsize=(12, 3 * len(conds_present)), squeeze=False)
        for ax, cond_name in zip(axes[:, 0], conds_present):
            toks, weights, labels = chart_data[(item_id, cond_name)]
            colors = [SEGMENT_COLORS[l] for l in labels]
            ax.bar(range(len(toks)), weights, color=colors)
            ax.set_xticks(range(len(toks)))
            ax.set_xticklabels([t.strip() or "·" for t in toks], rotation=60, ha="right", fontsize=8)
            ax.set_title(cond_name)
            ax.set_ylabel("attention\n(sink excluded, renorm.)")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SEGMENT_COLORS.values()]
        fig.legend(handles, SEGMENT_COLORS.keys(), loc="upper right")
        fig.suptitle(f"item_id={item_id}")
        fig.tight_layout()
        fig.savefig(f"{charts_dir}/item{item_id}.png", dpi=150)
        plt.close(fig)
    print(f"\nSaved content-only attention bar charts to {charts_dir}/")
