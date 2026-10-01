"""Shared evaluation metrics for central tendency and disagreement.

Every experiment script (Step 3 baseline, Step 4 zero/few-shot LLM, Step 5
persona ensembling, ...) should import from this module rather than
reimplementing metrics, so numbers stay comparable across the results table.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr

RATING_SCALE = (0, 1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Central tendency: point-estimate predictions (mean/median rating per item)
# ---------------------------------------------------------------------------

def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pearson_r(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(pearsonr(y_true, y_pred)[0])


def spearman_r(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(spearmanr(y_true, y_pred)[0])


def evaluate_central_tendency(y_true, y_pred) -> dict:
    """MAE, RMSE, Pearson r, Spearman rho — for either the mean-rating task
    or the disagreement task when disagreement is framed as scalar regression
    (e.g. predicting entropy directly, Step 3.3)."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "pearson_r": pearson_r(y_true, y_pred),
        "spearman_r": spearman_r(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Disagreement: metrics over per-item rating distributions (0-4)
# ---------------------------------------------------------------------------

def ratings_to_distribution(ratings, scale=RATING_SCALE, smoothing: float = 0.0) -> np.ndarray:
    """Empirical probability distribution over `scale` from raw ratings.
    Values outside `scale` (e.g. -1 "didn't understand") are dropped."""
    ratings = np.asarray([r for r in ratings if r in scale])
    counts = np.array([np.sum(ratings == v) for v in scale], dtype=float)
    counts += smoothing
    total = counts.sum()
    if total == 0:
        return np.full(len(scale), 1.0 / len(scale))
    return counts / total


def refusal_rate(ratings) -> float:
    """Fraction of raw ratings equal to -1 ('didn't understand'). Track this
    alongside entropy rather than folding it in silently — it clusters by
    country (UK/Canada/NZ ~11-13%) and is a comprehension confound, not
    genuine offensiveness disagreement."""
    ratings = np.asarray(ratings)
    if len(ratings) == 0:
        return float("nan")
    return float(np.mean(ratings == -1))


def entropy(p, base: float = 2.0) -> float:
    """Shannon entropy of a discrete distribution (array-like, sums to 1)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)) / np.log(base))


def kl_divergence(p, q, eps: float = 1e-12) -> float:
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    mask = p > 0
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def jsd(p, q, base: float = 2.0) -> float:
    """Jensen-Shannon divergence: symmetric, bounded [0, 1] at base=2."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return float((0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)) / np.log(base))


def cross_entropy(p_true, q_pred, eps: float = 1e-12) -> float:
    """CE(p, q) = -sum p*log(q): score a predicted distribution q against
    the empirical human distribution p. Lower is better."""
    p, q = np.asarray(p_true, dtype=float), np.asarray(q_pred, dtype=float)
    q = np.clip(q, eps, 1.0)
    return float(-np.sum(p * np.log(q)))


def evaluate_disagreement_distributional(true_dists, pred_dists) -> dict:
    """When disagreement is predicted as a full distribution over the rating
    scale rather than a scalar — mean entropy gap, JSD, cross-entropy."""
    true_dists = np.asarray(true_dists, dtype=float)
    pred_dists = np.asarray(pred_dists, dtype=float)
    true_entropy = np.array([entropy(p) for p in true_dists])
    pred_entropy = np.array([entropy(q) for q in pred_dists])
    jsds = np.array([jsd(p, q) for p, q in zip(true_dists, pred_dists)])
    ces = np.array([cross_entropy(p, q) for p, q in zip(true_dists, pred_dists)])
    return {
        "mean_true_entropy": float(np.mean(true_entropy)),
        "mean_pred_entropy": float(np.mean(pred_entropy)),
        "entropy_mae": float(np.mean(np.abs(true_entropy - pred_entropy))),
        "mean_jsd": float(np.mean(jsds)),
        "mean_cross_entropy": float(np.mean(ces)),
    }


# ---------------------------------------------------------------------------
# Grouped breakdowns (category / region / disagreement-level), used by the
# residual checks in Step 3.7 and Step 4.6
# ---------------------------------------------------------------------------

def disagreement_bucket(entropy_values, labels=("low", "medium", "high")) -> np.ndarray:
    """Tertile-bucket a vector of per-item entropy values into low/medium/high
    human disagreement, for breaking out error by disagreement level."""
    entropy_values = np.asarray(entropy_values, dtype=float)
    edges = np.quantile(entropy_values, [1 / 3, 2 / 3])
    idx = np.digitize(entropy_values, edges)
    return np.asarray(labels)[idx]


def evaluate_by_group(y_true, y_pred, groups, metric_fn=evaluate_central_tendency) -> dict:
    """Apply `metric_fn` separately within each distinct value of `groups`
    (e.g. item category, region, or a disagreement_bucket output)."""
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    out = {}
    for g in np.unique(groups):
        mask = groups == g
        out[str(g)] = metric_fn(y_true[mask], y_pred[mask])
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, size=200).astype(float)
    y_pred = y_true + rng.normal(0, 0.5, size=200)
    print("central tendency:", evaluate_central_tendency(y_true, y_pred))

    true_dists = np.array([ratings_to_distribution(rng.integers(0, 5, size=8)) for _ in range(50)])
    pred_dists = np.array([ratings_to_distribution(rng.integers(0, 5, size=8)) for _ in range(50)])
    print("disagreement:", evaluate_disagreement_distributional(true_dists, pred_dists))

    buckets = disagreement_bucket([entropy(p) for p in true_dists])
    print("group breakdown:", evaluate_by_group(y_true[:50], y_pred[:50], buckets))
