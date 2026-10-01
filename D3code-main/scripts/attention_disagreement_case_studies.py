"""
Case studies grounded in REAL annotator disagreement, not hypothetical framing.

Where attention_probe_detailed.py stratified 30 items evenly across D3CODE's
3 categories, this instead picks items where D3CODE's own raters split along a
real demographic line -- e.g. men and women rating the same text very
differently -- using the dataset's own rater metadata (Gender, Region), not
an assumption about what "should" cause disagreement.

For each such item we then ask the exact question your supervisor posed:
does the model's classification change once we tell it the matching
demographic identity, compared to the plain baseline prompt? And we extract
the same attention information (F1-F3 from the report) plus the bertviz
attention-flow visualization and the content-only bar chart, reusing the
exact same code as attention_probe_detailed.py, applied to these specific
human-disagreement items instead of the stratified sample.

Two axes are used, both grounded in real D3CODE rater fields:
- Gender: rater-reported Man vs Woman (D3CODE's own Gender field)
- Region: rater-reported Western Europe vs Arab Culture (D3CODE's own Region
  field -- these are the exact category strings D3CODE uses, matching the
  demographic conditions already defined in attention_probe.py)
Religion is NOT included here: D3CODE has no religion field for raters, so
there's no real annotator-disagreement signal to ground it in (this was
already flagged when religion_christian/religion_muslim were first added --
those conditions are a hypothesis probe, not a validated D3CODE contrast).

Run as a batch job (needs a GPU) -- see
attention_disagreement_case_studies_job.sbatch.
"""
import json
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bertviz import head_view

from attention_probe import CONDITIONS, find_token_span, find_span_by_delimiters, build_prompt
from zeroshot_two_datasets import PROMPT

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
D3_ITEMS_CSV = "../dataset/d3-items.csv"
D3_RATINGS_CSV = "../dataset/d3-ratings.csv"
D3_RATERS_CSV = "../dataset/d3-raters.csv"
D3_SPLITS_JSON = "../splits.json"
MAX_LENGTH = 1024
TOP_K_PER_AXIS = 3
MIN_RATERS_PER_GROUP = 5  # avoid picking items where the "disagreement" is just 2-3 noisy ratings

RESULTS_DIR = "results/disagreement_case_studies"
FIGURES_DIR = "figures/disagreement_case_studies"
CASES_OUT_PATH = f"{RESULTS_DIR}/disagreement_case_items.csv"
RESULTS_OUT_PATH = f"{RESULTS_DIR}/disagreement_case_results.csv"
INSTRUCTION_TEXT = "Is this text offensive? Answer with only 0 (no) or 1 (yes)."
TEXT_PREFIX, TEXT_SUFFIX = PROMPT.split("{text}")

AXES = {
    "gender": {
        "rater_col": "Gender", "val_a": "Man", "val_b": "Woman",
        "conditions": {"baseline": None, "gender_man": CONDITIONS["gender_man"], "gender_woman": CONDITIONS["gender_woman"]},
    },
    "region": {
        "rater_col": "Region", "val_a": "Western Europe", "val_b": "Arab Culture",
        "conditions": {"baseline": None, "region_western_europe": CONDITIONS["region_western_europe"], "region_arab_culture": CONDITIONS["region_arab_culture"]},
    },
}


def select_disagreement_items():
    """For each axis, find items in the frozen test split where the two rater
    groups' mean rating_binary differs most, requiring at least
    MIN_RATERS_PER_GROUP ratings from each group so the gap isn't just noise
    from 2-3 raters. Returns a DataFrame: item_id, axis, text, category,
    mean_a, mean_b, n_a, n_b, gap."""
    items = pd.read_csv(D3_ITEMS_CSV)
    ratings = pd.read_csv(D3_RATINGS_CSV)
    raters = pd.read_csv(D3_RATERS_CSV)
    with open(D3_SPLITS_JSON) as f:
        test_ids = set(json.load(f)["test"])

    merged = ratings.merge(raters[["rater_id", "Gender", "Region"]], on="rater_id", how="left")
    merged = merged.dropna(subset=["rating_binary"])
    merged = merged[merged["item_id"].isin(test_ids)]

    rows = []
    for axis, spec in AXES.items():
        col, val_a, val_b = spec["rater_col"], spec["val_a"], spec["val_b"]
        sub = merged[merged[col].isin([val_a, val_b])]
        g = sub.groupby(["item_id", col])["rating_binary"].agg(["mean", "count"]).unstack()
        g.columns = ["_".join(c) for c in g.columns]
        need = [f"mean_{val_a}", f"mean_{val_b}", f"count_{val_a}", f"count_{val_b}"]
        g = g.dropna(subset=need)
        g = g[(g[f"count_{val_a}"] >= MIN_RATERS_PER_GROUP) & (g[f"count_{val_b}"] >= MIN_RATERS_PER_GROUP)]
        g["gap"] = (g[f"mean_{val_a}"] - g[f"mean_{val_b}"]).abs()
        top = g.sort_values("gap", ascending=False).head(TOP_K_PER_AXIS)
        for item_id, row in top.iterrows():
            item_row = items[items["item_id"] == item_id]
            if len(item_row) == 0:
                continue
            rows.append({
                "item_id": item_id, "axis": axis,
                "text": item_row.iloc[0]["text"], "category": item_row.iloc[0]["category"],
                f"mean_{val_a}": row[f"mean_{val_a}"], f"mean_{val_b}": row[f"mean_{val_b}"],
                f"n_{val_a}": row[f"count_{val_a}"], f"n_{val_b}": row[f"count_{val_b}"],
                "gap": row["gap"],
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cases = select_disagreement_items()
    cases.to_csv(CASES_OUT_PATH, index=False)
    print(f"Selected {len(cases)} human-disagreement items:")
    print(cases[["item_id", "axis", "gap", "category"]].to_string(index=False))

    print(f"\nLoading model: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    )
    model.eval()

    zero_id = tok.encode("0", add_special_tokens=False)[0]
    one_id = tok.encode("1", add_special_tokens=False)[0]

    result_rows = []
    chart_data_by_item = {}  # item_id -> {condition: (tokens, weights, labels)}
    heatmaps_dir = f"{FIGURES_DIR}/attention_heatmaps_case_studies"
    charts_dir = f"{FIGURES_DIR}/content_attention_charts_case_studies"
    os.makedirs(heatmaps_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    with torch.no_grad():
        for _, case in cases.iterrows():
            conditions = AXES[case["axis"]]["conditions"]
            for cond_name, cond_prefix in conditions.items():
                prompt = build_prompt(tok, cond_prefix, case["text"])
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

                attn_stack = torch.stack(out.attentions, dim=0)
                attn = attn_stack[:, 0, :, -1, :].float().mean(dim=(0, 1)).cpu().numpy()

                input_ids = inputs["input_ids"][0].tolist()
                tokens = [tok.decode([tid]) for tid in input_ids]

                demo_span = find_token_span(tokens, cond_prefix) if cond_prefix else None
                instr_span = find_token_span(tokens, INSTRUCTION_TEXT)
                text_span = find_span_by_delimiters(tokens, TEXT_PREFIX, TEXT_SUFFIX)

                if instr_span is None or text_span is None:
                    print(f"  WARNING: span not found for item_id={case['item_id']} condition={cond_name} "
                          f"(instr_span={instr_span}, text_span={text_span}) -- skipping attention breakdown for this row")
                    pct_demo = pct_instr = pct_text = pct_other = np.nan
                else:
                    pct_demo = float(attn[demo_span[0]: demo_span[1] + 1].sum()) if demo_span else np.nan
                    pct_instr = float(attn[instr_span[0]: instr_span[1] + 1].sum())
                    pct_text = float(attn[text_span[0]: text_span[1] + 1].sum())
                    pct_other = 1.0 - pct_text - pct_instr - (pct_demo if not np.isnan(pct_demo) else 0.0)

                content_mass = 1.0 - pct_other if not np.isnan(pct_other) else np.nan
                pct_demo_norm = (pct_demo / content_mass) if (content_mass and content_mass > 0 and not np.isnan(pct_demo)) else np.nan
                pct_instr_norm = (pct_instr / content_mass) if (content_mass and content_mass > 0) else np.nan
                pct_text_norm = (pct_text / content_mass) if (content_mass and content_mass > 0) else np.nan

                result_rows.append({
                    "item_id": case["item_id"], "axis": case["axis"], "condition": cond_name,
                    "category": case["category"], "predicted": predicted,
                    "pct_demographic": pct_demo, "pct_instruction": pct_instr,
                    "pct_item_text": pct_text, "pct_other_scaffolding": pct_other,
                    "pct_demographic_norm": pct_demo_norm, "pct_instruction_norm": pct_instr_norm,
                    "pct_item_text_norm": pct_text_norm,
                })

                # bertviz attention-flow visualization -- same as attention_probe_detailed.py
                html = head_view(tuple(a.cpu() for a in out.attentions), tokens, html_action="return")
                with open(f"{heatmaps_dir}/item{case['item_id']}_{cond_name}.html", "w") as f:
                    f.write(html.data)

                # content-only bar chart data (sink excluded, renormalized) -- collected
                # per item below, one chart per item covering all its conditions
                if not np.isnan(pct_other):
                    spans = [("demographic", demo_span), ("instruction", instr_span), ("item_text", text_span)]
                    content_idx = sorted({i for _, span in spans if span for i in range(span[0], span[1] + 1)})
                    idx_to_label = {i: label for label, span in spans if span for i in range(span[0], span[1] + 1)}
                    content_attn = np.array([attn[i] for i in content_idx])
                    content_attn_norm = content_attn / content_attn.sum() if content_attn.sum() > 0 else content_attn
                    chart_data_by_item.setdefault(case["item_id"], {})[cond_name] = (
                        [tokens[i] for i in content_idx], content_attn_norm,
                        [idx_to_label[i] for i in content_idx],
                    )

    results_df = pd.DataFrame(result_rows)

    base_pred = results_df[results_df.condition == "baseline"][["item_id", "predicted"]].rename(columns={"predicted": "baseline_pred"})
    results_df = results_df.merge(base_pred, on="item_id")
    results_df["flipped_vs_baseline"] = results_df["predicted"] != results_df["baseline_pred"]
    results_df.to_csv(RESULTS_OUT_PATH, index=False)
    print(f"\nSaved {CASES_OUT_PATH} and {RESULTS_OUT_PATH}")

    print("\n=== Per-item results: human disagreement vs. model behavior ===")
    for _, case in cases.iterrows():
        sub = results_df[results_df.item_id == case["item_id"]]
        print(f"\nitem_id={case['item_id']}  axis={case['axis']}  category={case['category']}  "
              f"human_gap={case['gap']:.2f}")
        print(f"  text={case['text'][:100]!r}")
        for _, r in sub.iterrows():
            flip_note = "  <-- FLIPPED vs baseline" if r["flipped_vs_baseline"] and r["condition"] != "baseline" else ""
            print(f"  [{r['condition']:24s}] predicted={r['predicted']}  "
                  f"pct_demographic_norm={r['pct_demographic_norm']:.4f}{flip_note}")

    print(f"\nSaved bertviz attention-flow heatmaps to {heatmaps_dir}/")

    # content-only bar charts, one per item (all its conditions stacked), same style
    # as attention_probe_detailed.py's content_attention_charts/
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SEGMENT_COLORS = {"demographic": "tab:red", "instruction": "tab:blue", "item_text": "tab:green"}
    for _, case in cases.iterrows():
        chart_data = chart_data_by_item.get(case["item_id"])
        if not chart_data:
            continue
        conds_present = list(chart_data.keys())
        fig, axes = plt.subplots(len(conds_present), 1, figsize=(12, 3 * len(conds_present)), squeeze=False)
        for ax, cond_name in zip(axes[:, 0], conds_present):
            toks, weights, labels = chart_data[cond_name]
            colors = [SEGMENT_COLORS[l] for l in labels]
            ax.bar(range(len(toks)), weights, color=colors)
            ax.set_xticks(range(len(toks)))
            ax.set_xticklabels([t.strip() or "·" for t in toks], rotation=60, ha="right", fontsize=8)
            ax.set_title(cond_name)
            ax.set_ylabel("attention\n(sink excluded, renorm.)")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SEGMENT_COLORS.values()]
        fig.legend(handles, SEGMENT_COLORS.keys(), loc="upper right")
        fig.suptitle(f"item_id={case['item_id']}  (axis={case['axis']}, human disagreement gap={case['gap']:.2f})")
        fig.tight_layout()
        fig.savefig(f"{charts_dir}/item{case['item_id']}.png", dpi=150)
        plt.close(fig)
    print(f"Saved content-only attention bar charts to {charts_dir}/")
