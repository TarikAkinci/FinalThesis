"""
Simple zero-shot classification performance across TWO datasets: D3CODE and
Measuring Hate Speech (MHS). Per supervisor instruction: no persona prompting,
no few-shot, no prompt engineering -- one fixed minimal prompt, applied
identically to both datasets, 500 items sampled from each.

Real generate() call per item (same fixed prompt, nothing forced). From that
same call, output_scores=True also gives us P(first token = "1") for free --
a continuous confidence score, with no extra forward pass. That score lets us
pick a decision threshold that maximizes macro F1 (average of F1 for each
class), instead of accepting whatever operating point the model's own greedy
answer happens to produce. The threshold is tuned on a small held-out sample
DISJOINT from the reported 500-item evaluation sample, so the reported number
isn't tuned on the same data it's evaluated on.

Run as a batch job (see zeroshot_two_datasets_job.sbatch).
"""
import json
import os
import numpy as np
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
# sklearn is imported lazily inside best_macro_f1_threshold()/evaluate() below, not here:
# other scripts (master_grid.py, high_disagreement_items.py) import PROMPT and
# prepare_d3code() from this module and never call those two functions, so they
# shouldn't need sklearn installed just to import this file.

# Comma-separated list, run sequentially in one job so both models score the
# exact same samples. Override with e.g.
# `MODEL_NAMES=Qwen/Qwen2.5-7B-Instruct python zeroshot_two_datasets.py` for a single model.
MODEL_NAMES = [m.strip() for m in os.environ.get(
    "MODEL_NAMES", "meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen2.5-7B-Instruct"
).split(",")]
N_SAMPLES = 500       # reported evaluation sample
N_TUNE = 150          # disjoint sample used only to pick the threshold
SEED = 42
MAX_NEW_TOKENS = 10

D3_ITEMS_CSV = "../dataset/d3-items.csv"
D3_RATINGS_CSV = "../dataset/d3-ratings.csv"
D3_SPLITS_JSON = "../splits.json"
D3_ITEM_ID_COL = "item_id"
D3_TEXT_COL = "text"
D3_RATINGS_ITEM_COL = "item_id"
D3_RATINGS_VALUE_COL = "rating_raw"
D3_BINARY_VALUE_COL = "rating_binary"

MHS_CSV = "../dataset/mhs.csv"
MHS_TEXT_COL = "text"
MHS_LABEL_COL = "hate_speech_score"
MHS_BINARY_THRESHOLD = 0.5


PROMPT = "Text: \"{text}\"\nIs this text offensive? Answer with only 0 (no) or 1 (yes)."
MIN_POSITIVE_FRAC = 0.20  # guarantee this floor on the EVAL sample, not a fixed target


def stratified_sample(data, n_samples, min_positive_frac, seed):
    """`data` must already be deduped by text. Samples n_samples items with AT
    LEAST min_positive_frac positive class -- a floor, not a fixed target. If
    the pool already clears the floor, plain random sampling preserves the
    real rate rather than throwing away signal to force an exact percentage."""
    if data["label"].mean() >= min_positive_frac:
        return data.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    pos, neg = data[data["label"] == 1], data[data["label"] == 0]
    n_pos = min(int(n_samples * min_positive_frac), len(pos))
    n_neg = n_samples - n_pos
    if n_neg > len(neg):
        raise ValueError(f"not enough negative items after dedup: need {n_neg}, have {len(neg)}")

    sampled = pd.concat([
        pos.sample(n=n_pos, random_state=seed),
        neg.sample(n=n_neg, random_state=seed),
    ])
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle, don't leave positives first


def prepare_eval_and_tune(data, n_eval, n_tune, min_positive_frac, seed):
    """Dedup by text (comments must be unique), draw the reported eval sample
    (>= min_positive_frac positive), then a DISJOINT tuning sample from what's
    left -- plain random, only used to pick a threshold, never reported."""
    data = data.drop_duplicates(subset="text").reset_index(drop=True)
    eval_sample = stratified_sample(data, n_eval, min_positive_frac, seed)
    remaining = data[~data["text"].isin(eval_sample["text"])].reset_index(drop=True)
    tune_sample = remaining.sample(n=min(n_tune, len(remaining)), random_state=seed).reset_index(drop=True)
    return eval_sample, tune_sample


def prepare_d3code():
    items = pd.read_csv(D3_ITEMS_CSV)
    ratings = pd.read_csv(D3_RATINGS_CSV)
    with open(D3_SPLITS_JSON) as f:
        test_ids = set(json.load(f)["test"])

    majority_label = ratings.dropna(subset=[D3_BINARY_VALUE_COL]) \
        .groupby(D3_RATINGS_ITEM_COL)[D3_BINARY_VALUE_COL].mean()
    data = items.merge(majority_label.rename("label_frac"), left_on=D3_ITEM_ID_COL, right_index=True)
    data = data[data[D3_ITEM_ID_COL].isin(test_ids)]  # frozen test split only, matches Step 3
    data["label"] = (data["label_frac"] >= 0.5).astype(int)
    data = data.rename(columns={D3_TEXT_COL: "text"})[["text", "label"]]
    return prepare_eval_and_tune(data, N_SAMPLES, N_TUNE, MIN_POSITIVE_FRAC, SEED)


def prepare_mhs():
    # MHS is post-level with multiple annotations per post -- aggregate first.
    raw = pd.read_csv(MHS_CSV)
    grouped = raw.groupby("comment_id" if "comment_id" in raw.columns else MHS_TEXT_COL).agg(
        text=(MHS_TEXT_COL, "first"),
        score=(MHS_LABEL_COL, "mean"),
    ).reset_index(drop=True)
    grouped["label"] = (grouped["score"] >= MHS_BINARY_THRESHOLD).astype(int)
    data = grouped[["text", "label"]]
    return prepare_eval_and_tune(data, N_SAMPLES, N_TUNE, MIN_POSITIVE_FRAC, SEED)


def parse_response(text):
    """Does the response start with 0 or 1? Anything else (hedges, refusals,
    explanations before the digit) is reported as malformed rather than
    silently coerced."""
    text = text.strip()
    if text.startswith("1"):
        return 1
    if text.startswith("0"):
        return 0
    return None


def run_with_scores(model, tok, zero_id, one_id, data, dataset_name, tag):
    """Real generate() per item, same fixed prompt. Also reads P(first token
    = "1") from that same call's output_scores -- no extra forward pass."""
    generated_texts, preds, prob_ones = [], [], []
    with torch.no_grad():
        for i, row in data.iterrows():
            messages = [{"role": "user", "content": PROMPT.format(text=row["text"])}]
            # enable_thinking=False matters for models like Qwen3 that default to a
            # <think>...</think> block first -- ignored by templates that don't use it
            # (e.g. Llama's), so safe to pass unconditionally across models.
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tok.eos_token_id, output_scores=True, return_dict_in_generate=True,
            )
            generated = tok.decode(out.sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            generated_texts.append(generated)
            preds.append(parse_response(generated))

            first_step_logits = out.scores[0][0]  # logits for the first generated token, full vocab
            prob_zero, prob_one = torch.softmax(first_step_logits[[zero_id, one_id]], dim=0)
            prob_ones.append(prob_one.item())

            if (i + 1) % 50 == 0:
                print(f"[{dataset_name}/{tag}] {i + 1}/{len(data)}")

    data = data.copy()
    data["generated"] = generated_texts
    data["predicted"] = preds
    data["prob_one"] = prob_ones
    return data


def best_macro_f1_threshold(y_true, scores):
    """Sweep thresholds on the continuous score, return the one maximizing
    macro F1. Called only on the held-out tuning set, never on eval data."""
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (scores >= t).astype(int)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def evaluate(y_true, y_pred, y_score, model_name, dataset_name, rule, threshold, n_malformed):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    # roc_auc_score needs the continuous score (prob_one), not the thresholded
    # 0/1 prediction -- it's threshold-independent by design, so it comes out
    # identical for the greedy_response and tuned_threshold rows of the same
    # model/dataset (same underlying scores, just two different cutoffs
    # applied downstream). That's expected, not a copy-paste bug.
    return {
        "model": model_name,
        "dataset": dataset_name,
        "rule": rule,
        "threshold": threshold,
        "n": len(y_true),
        "n_malformed": n_malformed,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred": float(np.mean(y_pred)),
    }


if __name__ == "__main__":
    # data prep doesn't depend on the model -- do it once so every model in
    # MODEL_NAMES scores the exact same eval + tune samples
    d3_eval, d3_tune = prepare_d3code()
    mhs_eval, mhs_tune = prepare_mhs()
    print(f"D3CODE eval: n={len(d3_eval)}, positive rate={d3_eval['label'].mean():.2f}  |  tune: n={len(d3_tune)}, positive rate={d3_tune['label'].mean():.2f}")
    print(f"MHS eval: n={len(mhs_eval)}, positive rate={mhs_eval['label'].mean():.2f}  |  tune: n={len(mhs_tune)}, positive rate={mhs_tune['label'].mean():.2f}")

    all_results = []
    for model_name in MODEL_NAMES:
        print(f"\n=== Loading model: {model_name} ===")
        model_slug = model_name.split("/")[-1]
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()

        zero_id = tok.encode("0", add_special_tokens=False)[0]
        one_id = tok.encode("1", add_special_tokens=False)[0]
        assert tok.decode([zero_id]) == "0" and tok.decode([one_id]) == "1", \
            f"token id mismatch: decode({zero_id})={tok.decode([zero_id])!r}, decode({one_id})={tok.decode([one_id])!r}"

        for dataset_name, eval_data, tune_data in [("d3code", d3_eval, d3_tune), ("mhs", mhs_eval, mhs_tune)]:
            tune_scored = run_with_scores(model, tok, zero_id, one_id, tune_data, dataset_name, "tune")
            threshold, tune_f1_macro = best_macro_f1_threshold(tune_scored["label"].values, tune_scored["prob_one"].values)
            print(f"[{dataset_name}] tuned threshold={threshold:.2f} (macro F1 on tune set={tune_f1_macro:.3f}, n={len(tune_data)})")

            eval_scored = run_with_scores(model, tok, zero_id, one_id, eval_data, dataset_name, "eval")
            os.makedirs("results/zeroshot", exist_ok=True)
            eval_scored.to_csv(f"results/zeroshot/predictions_{dataset_name}_{model_slug}_scored.csv", index=False)

            n_malformed = int(eval_scored["predicted"].isna().sum())
            naive_valid = eval_scored.dropna(subset=["predicted"])
            all_results.append(evaluate(
                naive_valid["label"], naive_valid["predicted"].astype(int), naive_valid["prob_one"],
                model_name, dataset_name, rule="greedy_response", threshold=None, n_malformed=n_malformed,
            ))

            tuned_pred = (eval_scored["prob_one"] >= threshold).astype(int)
            all_results.append(evaluate(
                eval_scored["label"], tuned_pred, eval_scored["prob_one"],
                model_name, dataset_name, rule="tuned_threshold", threshold=round(threshold, 2), n_malformed=0,
            ))

        del model, tok
        torch.cuda.empty_cache()

    new_results = pd.DataFrame(all_results)

    os.makedirs("results/zeroshot", exist_ok=True)
    out_path = "results/zeroshot/results_zeroshot_tuned.csv"
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        run_keys = set(zip(new_results["model"], new_results["dataset"], new_results["rule"]))
        existing = existing[~existing.apply(lambda r: (r["model"], r["dataset"], r["rule"]) in run_keys, axis=1)]
        results_df = pd.concat([existing, new_results], ignore_index=True)
    else:
        results_df = new_results
    results_df.to_csv(out_path, index=False)

    print("\n--- This run ---")
    print(new_results.to_string(index=False))
    print("\n--- All results so far (all models) ---")
    print(results_df.to_string(index=False))
