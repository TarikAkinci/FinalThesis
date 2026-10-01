"""
Training-free probing of attention head outputs, per supervisor instruction
("no training necessary"). For each (layer, head), compute a DiffMean
direction from the TRAIN split -- mean activation of high-disagreement items
minus mean of low-disagreement items, a plain subtraction, not a fitted
classifier -- then project TEST items onto that direction and measure
separability via AUC on the raw projection score. No logistic regression, no
cross-validation, no regularization search.

This matches Gurgurov et al.'s and FairSteer's own DiffMean/DSV computation
exactly, so the direction found here is directly reusable as a steering
vector later -- this step isn't just diagnostic, it produces what steering
needs too.

Uses the same per-head outputs extract_attention_heads.py saved
(attention_heads.npz). Fast, CPU-only, no GPU needed.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

IN_PATH = "attention_heads.npz"
OUT_PATH = "attention_head_diffmean_results.csv"
DIRECTIONS_OUT_PATH = "attention_head_directions.npz"  # actual vectors, for steering
TOP_K = 20


def diffmean_direction(X, y):
    """X: [n, head_dim], y: [n] binary (1 = high disagreement). Returns the
    (unnormalized) direction vector -- purely a difference of means."""
    return X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)


if __name__ == "__main__":
    data = np.load(IN_PATH)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    layers = data["layers"]
    num_heads = int(data["num_heads"])

    print(f"train: n={len(y_train)}, positive rate={y_train.mean():.1%}")
    print(f"test:  n={len(y_test)}, positive rate={y_test.mean():.1%}")
    print(f"{len(layers)} layers x {num_heads} heads = {len(layers) * num_heads} directions, no training\n")

    head_dim = X_train.shape[-1]
    all_directions = np.zeros((len(layers), num_heads, head_dim), dtype=np.float32)

    results = []
    for li, layer in enumerate(layers):
        for h in range(num_heads):
            X_tr, X_te = X_train[:, li, h, :], X_test[:, li, h, :]
            direction = diffmean_direction(X_tr, y_train)
            all_directions[li, h] = direction
            scores = X_te @ direction  # raw dot-product projection, no fitted classifier
            auc = roc_auc_score(y_test, scores)
            results.append({
                "layer": int(layer), "head": h,
                "test_auc": auc, "direction_norm": float(np.linalg.norm(direction)),
            })
        print(f"layer {layer:>2} done")

    results_df = pd.DataFrame(results).sort_values("test_auc", ascending=False)
    results_df.to_csv(OUT_PATH, index=False)
    np.savez(DIRECTIONS_OUT_PATH, directions=all_directions, layers=layers, num_heads=num_heads)
    print(f"Saved {DIRECTIONS_OUT_PATH} (shape {all_directions.shape}) -- the actual vectors, for steering")

    print(f"\nSaved {OUT_PATH}")
    print(f"\nTop {TOP_K} heads by test AUC (DiffMean projection, no training):")
    print(results_df.head(TOP_K).to_string(index=False))
    print(
        "\nNote: AUC below 0.5 for a head means its DiffMean direction "
        "(fixed from train data) doesn't separate test items in the "
        "expected direction -- a genuine result, not flipped/corrected."
    )
