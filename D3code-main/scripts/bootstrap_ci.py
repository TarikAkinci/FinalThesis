"""Bootstrap confidence intervals for the zero-shot metrics, reusing the
already-saved per-item predictions -- no model/GPU needed, just resampling."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

N_BOOT = 2000
SEED = 42

def bootstrap(path, dataset_name):
    df = pd.read_csv(path)
    rng = np.random.default_rng(SEED)
    n = len(df)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)  # resample with replacement
        y_true, y_pred = df["label"].values[idx], df["predicted"].values[idx]
        metrics["accuracy"].append(accuracy_score(y_true, y_pred))
        metrics["precision"].append(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"].append(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"].append(f1_score(y_true, y_pred, zero_division=0))

    print(f"\n{dataset_name} (n={n}, {N_BOOT} bootstrap resamples):")
    for name, vals in metrics.items():
        vals = np.array(vals)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        print(f"  {name:10s} point={vals.mean():.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")

if __name__ == "__main__":
    bootstrap("results/zeroshot/legacy/predictions_d3code_Llama-3.1-8B-Instruct.csv", "D3CODE")
    bootstrap("results/zeroshot/legacy/predictions_mhs_Llama-3.1-8B-Instruct.csv", "MHS")
