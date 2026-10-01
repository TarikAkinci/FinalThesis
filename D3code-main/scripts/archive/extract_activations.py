"""
Extract Llama-3.1-8B hidden-state activations at layers [11, 13, 26, 28, 31]
(Gurgurov et al.'s own findings for this exact model -- originally used for
political-ideology probing; here we test whether human DISAGREEMENT is
similarly linearly decodable) for D3CODE items.

Each item is wrapped in the SAME prompt used for zero-shot classification
(zeroshot_two_datasets.py's PROMPT), so we're probing the representation the
model actually uses when performing the offensiveness judgment, not a generic
text embedding. The feature per item/layer is the LAST TOKEN's hidden state --
same position the classification logits were read from throughout this
project.

Target: binary, top vs bottom tertile of per-item entropy (matches the binary
probe methodology the cited papers use directly). Tertile edges are computed
from the TRAIN split only and applied to both splits, so no test-set
information leaks into the label definition. "Medium" disagreement items are
dropped for a clean contrast.

hidden_states indexing: hidden_states[0] is the embedding output,
hidden_states[L] is the residual stream after transformer block L (1-indexed
in that sense) -- LAYERS below are indices into this tuple directly.

Run as a batch job (see extract_activations_job.sbatch) -- needs a GPU.
"""
import json
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import ratings_to_distribution, entropy
from zeroshot_two_datasets import PROMPT

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
LAYERS = [11, 13, 26, 28, 31]
D3_ITEMS_CSV = "../dataset/d3-items.csv"
D3_RATINGS_CSV = "../dataset/d3-ratings.csv"
D3_SPLITS_JSON = "../splits.json"
OUT_PATH = "activations.npz"
MAX_LENGTH = 512


def compute_entropy(ratings):
    return entropy(ratings_to_distribution(ratings))


def prepare_items():
    items = pd.read_csv(D3_ITEMS_CSV)
    ratings = pd.read_csv(D3_RATINGS_CSV)
    with open(D3_SPLITS_JSON) as f:
        splits = json.load(f)
    train_ids, test_ids = set(splits["train"]), set(splits["test"])

    ent = ratings.groupby("item_id")["rating_raw"].apply(compute_entropy)
    df = items.merge(ent.rename("entropy"), on="item_id", how="inner")
    df["split"] = df["item_id"].apply(lambda i: "train" if i in train_ids else ("test" if i in test_ids else None))
    df = df.dropna(subset=["split"]).copy()

    # tertile edges from TRAIN only, applied to both splits -- no test-set
    # information leaks into where the low/high boundary is drawn
    train_entropy = df.loc[df["split"] == "train", "entropy"]
    lo_edge, hi_edge = train_entropy.quantile([1 / 3, 2 / 3])
    print(f"tertile edges (from train): low<={lo_edge:.3f}, high>={hi_edge:.3f}")

    def bucket(e):
        if e <= lo_edge:
            return 0
        if e >= hi_edge:
            return 1
        return np.nan  # medium -- dropped

    df["label"] = df["entropy"].apply(bucket)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    for split_name in ["train", "test"]:
        sub = df[df["split"] == split_name]
        print(f"{split_name}: n={len(sub)}, positive (high disagreement) rate={sub['label'].mean():.1%}")
    return df


def build_prompt(tok, text):
    messages = [{"role": "user", "content": PROMPT.format(text=text)}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def extract(model, tok, texts):
    """One forward pass per item (no generation needed -- output_hidden_states
    gives every layer's activations directly). Returns
    [n_items, len(LAYERS), hidden_size]."""
    feats = np.zeros((len(texts), len(LAYERS), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for i, text in enumerate(texts):
            prompt = build_prompt(tok, text)
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
            out = model(**inputs, output_hidden_states=True)
            for j, layer in enumerate(LAYERS):
                feats[i, j] = out.hidden_states[layer][0, -1].float().cpu().numpy()
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(texts)}")
    return feats


if __name__ == "__main__":
    df = prepare_items()

    print(f"\nLoading model: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    print(f"\nExtracting train activations (n={len(df_train)})...")
    X_train = extract(model, tok, df_train["text"].tolist())
    print(f"Extracting test activations (n={len(df_test)})...")
    X_test = extract(model, tok, df_test["text"].tolist())

    np.savez(
        OUT_PATH,
        X_train=X_train, y_train=df_train["label"].values,
        X_test=X_test, y_test=df_test["label"].values,
        layers=np.array(LAYERS),
    )
    print(f"\nSaved to {OUT_PATH}")
