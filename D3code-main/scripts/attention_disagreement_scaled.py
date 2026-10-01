"""
Disagreement + placebo study, v3, per your supervisor's latest instruction:

1. Item selection changed from "500 items pooled from the whole D3CODE
   dataset, ranked by a gender- or region-specific rater gap" to "~100 items
   selected from WITHIN the seeded 500-item D3CODE eval sample, ranked by
   overall rating entropy" -- see high_disagreement_items.py for the full
   rationale (same sample as the headline zero-shot metrics, entropy over
   the full rating distribution reusing evaluate.py rather than a
   demographic-group-specific gap). Because selection is no longer axis-
   specific, there's no more "which axis does this item belong to" question:
   every one of the ~100 items gets ALL conditions (both genders, both
   regions, placebo, baseline), not a per-axis subset. This also removes the
   (item_id, axis) duplicate-key handling the previous version needed.

2. Same P(yes) logging as before (softmax over just the "0"/"1" logits,
   matching zeroshot_two_datasets.py's prob_one).

3. Same placebo condition as before ("You are someone who prefers tea over
   coffee.", your supervisor's own wording), run on every item alongside
   baseline and all 4 demographic conditions.

v3.1: item selection now also requires baseline P(yes) to be in a reasonable
band (not already pinned near 0 or 1), not just high entropy. Pure
entropy-based selection turned out to concentrate on items the model already
calls "offensive" with near-certainty (95/100 items, median confidence
0.999), leaving almost no room for any perturbation to move the prediction
further up -- confounding the whole demographic-vs-placebo probability-shift
comparison. See select_items_with_baseline_filter.py, which must be run
first; this script now loads its output via load_filtered_items() instead
of computing entropy-only selection itself.

Run as a batch job (needs a GPU) -- see attention_disagreement_scaled_job.sbatch.
"""
import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer
from bertviz import head_view

from attention_probe import CONDITIONS, find_token_span, find_span_by_delimiters, build_prompt
from zeroshot_two_datasets import PROMPT
from high_disagreement_items import load_filtered_items

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
MAX_LENGTH = 1024

RESULTS_DIR = "results/disagreement_scaled"
FIGURES_DIR = "figures/disagreement_scaled"
CASES_OUT_PATH = f"{RESULTS_DIR}/disagreement_scaled_items.csv"
RESULTS_OUT_PATH = f"{RESULTS_DIR}/disagreement_scaled_results.csv"
INSTRUCTION_TEXT = "Is this text offensive? Answer with only 0 (no) or 1 (yes)."
TEXT_PREFIX, TEXT_SUFFIX = PROMPT.split("{text}")

# Your supervisor's exact suggested wording, unrelated to demographics, similar length
# to the shorter demographic prefixes ("You are a man." / "You are Muslim.").
PLACEBO_TEXT = "You are someone who prefers tea over coffee."

ALL_CONDITIONS = {
    "baseline": None,
    "gender_man": CONDITIONS["gender_man"],
    "gender_woman": CONDITIONS["gender_woman"],
    "region_western_europe": CONDITIONS["region_western_europe"],
    "region_arab_culture": CONDITIONS["region_arab_culture"],
    "placebo": PLACEBO_TEXT,
}
DEMOGRAPHIC_CONDITIONS = ["gender_man", "gender_woman", "region_western_europe", "region_arab_culture"]
N_ILLUSTRATIVE = 3  # top-entropy items get a bertviz heatmap + content-attention chart

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # entropy-ranked AND baseline-probability-band-filtered (see
    # select_items_with_baseline_filter.py) -- run that script first
    cases = load_filtered_items()
    cases.to_csv(CASES_OUT_PATH, index=False)
    print(f"Selected {len(cases)} high-disagreement items from the seeded 500-item eval sample "
          f"(entropy range {cases.entropy.min():.3f}-{cases.entropy.max():.3f})")

    print(f"\nLoading model: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    )
    model.eval()

    zero_id = tok.encode("0", add_special_tokens=False)[0]
    one_id = tok.encode("1", add_special_tokens=False)[0]

    n_total = len(cases) * len(ALL_CONDITIONS)
    n_done = 0
    result_rows = []

    illustrative_items = set(cases.head(N_ILLUSTRATIVE)["item_id"])  # already sorted by entropy desc
    heatmap_data = {}
    chart_data_by_item = {}

    with torch.no_grad():
        for _, case in cases.iterrows():
            for cond_name, cond_prefix in ALL_CONDITIONS.items():
                prefix_type = "none" if cond_prefix is None else ("placebo" if cond_name == "placebo" else "demographic")

                prompt = build_prompt(tok, cond_prefix, case["text"])
                inputs = tok(
                    prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH,
                    add_special_tokens=False,
                ).to(model.device)
                out = model(**inputs, output_attentions=True)

                # No .generate()/temperature involved: a single deterministic forward
                # pass. predicted label via direct logit comparison (argmax), and
                # prob_yes via the SAME restricted softmax formula used in
                # zeroshot_two_datasets.py (softmax over just the "0"/"1" logits, not
                # full vocab), for consistency with the rest of the project.
                logits = out.logits[0, -1]
                predicted = "1" if logits[one_id] > logits[zero_id] else "0"
                _, prob_one = torch.softmax(logits[[zero_id, one_id]].float(), dim=0)
                prob_yes = prob_one.item()

                attn_stack = torch.stack(out.attentions, dim=0)
                attn = attn_stack[:, 0, :, -1, :].float().mean(dim=(0, 1)).cpu().numpy()

                input_ids = inputs["input_ids"][0].tolist()
                tokens = [tok.decode([tid]) for tid in input_ids]

                demo_span = find_token_span(tokens, cond_prefix) if cond_prefix else None
                instr_span = find_token_span(tokens, INSTRUCTION_TEXT)
                text_span = find_span_by_delimiters(tokens, TEXT_PREFIX, TEXT_SUFFIX)

                if instr_span is None or text_span is None:
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
                    "item_id": case["item_id"], "condition": cond_name,
                    "prefix_type": prefix_type, "category": case["category"], "entropy": case["entropy"],
                    "predicted": predicted, "prob_yes": prob_yes,
                    "pct_demographic": pct_demo, "pct_instruction": pct_instr,
                    "pct_item_text": pct_text, "pct_other_scaffolding": pct_other,
                    "pct_demographic_norm": pct_demo_norm, "pct_instruction_norm": pct_instr_norm,
                    "pct_item_text_norm": pct_text_norm,
                })

                if case["item_id"] in illustrative_items:
                    heatmap_data[(case["item_id"], cond_name)] = (tuple(a.cpu() for a in out.attentions), tokens)
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

                n_done += 1
                if n_done % 100 == 0:
                    print(f"  {n_done}/{n_total}")

    results_df = pd.DataFrame(result_rows)
    base = results_df[results_df.condition == "baseline"][["item_id", "predicted", "prob_yes"]].rename(
        columns={"predicted": "baseline_pred", "prob_yes": "baseline_prob_yes"})
    results_df = results_df.merge(base, on="item_id", validate="many_to_one")
    results_df["flipped_vs_baseline"] = results_df["predicted"] != results_df["baseline_pred"]
    results_df["delta_prob_yes"] = results_df["prob_yes"] - results_df["baseline_prob_yes"]
    results_df.to_csv(RESULTS_OUT_PATH, index=False)
    print(f"\nSaved {CASES_OUT_PATH} and {RESULTS_OUT_PATH}")

    bad = results_df[results_df["pct_item_text"].isna() | results_df["pct_instruction"].isna()]
    print(f"\nRows with unfound span: {len(bad)}")
    if len(bad):
        print(bad[["item_id", "condition"]].to_string(index=False))

    # --- summary analysis ---
    non_baseline = results_df[results_df.condition != "baseline"]
    print(f"\n=== Flip rate by condition (n={len(cases)} items) ===")
    print(non_baseline.groupby("condition")["flipped_vs_baseline"].agg(["mean", "sum", "count"]))

    print("\n=== Does the SIZE of the item's overall rating entropy predict a bigger model shift? (Spearman) ===")
    demo_rows = results_df[results_df.prefix_type == "demographic"]
    per_item = demo_rows.groupby("item_id").agg(
        entropy=("entropy", "first"),
        mean_abs_delta_prob_yes=("delta_prob_yes", lambda s: s.abs().mean()),
        mean_demo_norm=("pct_demographic_norm", "mean"),
    ).dropna()
    r1, p1 = spearmanr(per_item["entropy"], per_item["mean_abs_delta_prob_yes"])
    r2, p2 = spearmanr(per_item["entropy"], per_item["mean_demo_norm"])
    print(f"  (n={len(per_item)}): entropy vs |delta P(yes)|: rho={r1:+.3f} p={p1:.4g}   "
          f"entropy vs pct_demographic_norm: rho={r2:+.3f} p={p2:.4g}")

    print("\n=== Mean |delta P(yes)| vs baseline, by condition ===")
    print(non_baseline.assign(abs_delta=non_baseline["delta_prob_yes"].abs())
          .groupby("condition")["abs_delta"].agg(["mean", "std"]))

    print("\n=== Placebo check: paired Wilcoxon, each demographic condition vs placebo, pct_demographic_norm ===")
    placebo_norm = results_df[results_df.condition == "placebo"].set_index("item_id")["pct_demographic_norm"]
    for cond in DEMOGRAPHIC_CONDITIONS:
        demo = results_df[results_df.condition == cond].set_index("item_id")["pct_demographic_norm"]
        merged = pd.DataFrame({"demo": demo, "placebo": placebo_norm}).dropna()
        diff = merged["demo"] - merged["placebo"]
        if len(diff) >= 5 and not (diff == 0).all():
            stat, p = wilcoxon(diff)
            print(f"  {cond:24s}: mean_demo={merged['demo'].mean():.4f}  mean_placebo={merged['placebo'].mean():.4f}  "
                  f"mean_diff={diff.mean():+.4f}  frac_demo_higher={(diff > 0).mean():.2f}  n={len(diff)}  p={p:.4g}")
        else:
            print(f"  {cond:24s}: insufficient data (n={len(diff)})")

    print("\n=== Placebo check: paired Wilcoxon, each demographic condition vs placebo, |delta P(yes)| ===")
    placebo_delta = results_df[results_df.condition == "placebo"].set_index("item_id")["delta_prob_yes"].abs()
    for cond in DEMOGRAPHIC_CONDITIONS:
        demo = results_df[results_df.condition == cond].set_index("item_id")["delta_prob_yes"].abs()
        merged = pd.DataFrame({"demo": demo, "placebo": placebo_delta}).dropna()
        diff = merged["demo"] - merged["placebo"]
        if len(diff) >= 5 and not (diff == 0).all():
            stat, p = wilcoxon(diff)
            print(f"  {cond:24s}: mean_abs_delta_demo={merged['demo'].mean():.4f}  "
                  f"mean_abs_delta_placebo={merged['placebo'].mean():.4f}  "
                  f"mean_diff={diff.mean():+.4f}  n={len(diff)}  p={p:.4g}")
        else:
            print(f"  {cond:24s}: insufficient data (n={len(diff)})")

    print("\n=== Direction check: how many items push toward 'offensive' vs 'not offensive', by condition ===")
    print(non_baseline.groupby("condition")["delta_prob_yes"].agg(
        n_positive=lambda s: (s > 0).sum(), n_negative=lambda s: (s < 0).sum(), mean="mean"))

    # illustrative bertviz heatmaps + content-only bar charts, top-entropy items only
    heatmaps_dir = f"{FIGURES_DIR}/attention_heatmaps_scaled"
    os.makedirs(heatmaps_dir, exist_ok=True)
    for (item_id, cond_name), (attentions, tokens) in heatmap_data.items():
        html = head_view(attentions, tokens, html_action="return")
        with open(f"{heatmaps_dir}/item{item_id}_{cond_name}.html", "w") as f:
            f.write(html.data)
    print(f"\nSaved {len(heatmap_data)} illustrative attention heatmaps to {heatmaps_dir}/")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SEGMENT_COLORS = {"demographic": "tab:red", "instruction": "tab:blue", "item_text": "tab:green"}
    charts_dir = f"{FIGURES_DIR}/content_attention_charts_scaled"
    os.makedirs(charts_dir, exist_ok=True)
    for item_id, chart_data in chart_data_by_item.items():
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
        fig.suptitle(f"item_id={item_id}")
        fig.tight_layout()
        fig.savefig(f"{charts_dir}/item{item_id}.png", dpi=150)
        plt.close(fig)
    print(f"Saved {len(chart_data_by_item)} illustrative content-attention charts to {charts_dir}/")
