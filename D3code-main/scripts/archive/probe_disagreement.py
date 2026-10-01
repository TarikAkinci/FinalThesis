"""
Train a linear probe per layer on the activations extract_activations.py
saved, to test whether human disagreement (high vs low entropy) is linearly
decodable from Llama-3.1-8B's hidden states -- the same binary-probe
methodology Gurgurov et al. used for political ideology, applied here to a
different construct.

Fast, CPU-only, no GPU needed -- run this after extract_activations.py,
either on the login node or after syncing activations.npz back locally.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

IN_PATH = "activations.npz"
OUT_PATH = "probe_results.csv"

if __name__ == "__main__":
    data = np.load(IN_PATH)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    layers = data["layers"]

    print(f"train: n={len(y_train)}, positive rate={y_train.mean():.1%}")
    print(f"test:  n={len(y_test)}, positive rate={y_test.mean():.1%}")
    print()

    results = []
    for i, layer in enumerate(layers):
        # 4096 features but ~2000 training examples -- fewer examples than
        # dimensions, so a fixed C=1.0 (LogisticRegression's default)
        # overfits badly (verified: gave perfect train acc, chance-level test
        # AUC, on a synthetic sanity check before this was ever run on real
        # data). LogisticRegressionCV picks C via internal cross-validation
        # on the training set only, so no test-set leakage.
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(Cs=np.logspace(-4, 1, 10), cv=5, max_iter=2000, scoring="roc_auc"),
        )
        probe.fit(X_train[:, i, :], y_train)

        train_acc = accuracy_score(y_train, probe.predict(X_train[:, i, :]))
        test_acc = accuracy_score(y_test, probe.predict(X_test[:, i, :]))
        test_auc = roc_auc_score(y_test, probe.predict_proba(X_test[:, i, :])[:, 1])

        print(f"layer {layer:>2}: train acc={train_acc:.3f}  test acc={test_acc:.3f}  test AUC={test_auc:.3f}")
        results.append({"layer": int(layer), "train_acc": train_acc, "test_acc": test_acc, "test_auc": test_auc})

    results_df = pd.DataFrame(results).sort_values("test_auc", ascending=False)
    results_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH}, best layer by test AUC:")
    print(results_df.iloc[0].to_string())
