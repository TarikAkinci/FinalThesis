"""
Shared helpers for the post-hoc analysis scripts (analysis_*.py). CPU only.

Loading conventions:
  - master_grid_<MODEL>.csv          one row per (item, variant, condition)
  - master_grid_perhead_<MODEL>.npz  attention (rows, layers, heads, 4 segments)
                                     at the decision token, rows aligned to the
                                     index arrays stored in the same file
Older runs lack demo_span_tokens / matched_placebo; the loaders fill them from
phrase_tokens and the placebo_tok{N} naming convention so old and new outputs
go through the same code.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

SEG_DEMO, SEG_INSTR, SEG_TEXT, SEG_OTHER = 0, 1, 2, 3
MATCHED_DEMOGRAPHICS = ["religion_muslim", "gender_woman",
                        "region_western_europe", "region_arab_culture"]


def grid_csv_paths(results_dir):
    return sorted(p for p in glob.glob(os.path.join(results_dir, "master_grid_*.csv"))
                  if not any(k in os.path.basename(p) for k in ("layerwise", "items", "pairing")))


def model_tag_from_path(path):
    base = os.path.basename(path)
    for prefix in ("master_grid_perhead_", "master_grid_layerwise_", "master_grid_items_", "master_grid_"):
        if base.startswith(prefix):
            return base[len(prefix):].split(".")[0]
    raise ValueError(path)


def _fill_matched_placebo(df):
    if "matched_placebo" not in df.columns:
        df["matched_placebo"] = ""
    df["matched_placebo"] = df["matched_placebo"].fillna("")
    for model, m in df.groupby("model"):
        present = set(m.condition)
        for cond in MATCHED_DEMOGRAPHICS:
            sel = (df.model == model) & (df.condition == cond)
            if not sel.any() or (df.loc[sel, "matched_placebo"] != "").any():
                continue
            name = f"placebo_tok{int(df.loc[sel, 'phrase_tokens'].iloc[0])}"
            if name in present:
                df.loc[sel, "matched_placebo"] = name
    return df


def load_grid(results_dir, model=None):
    """All master_grid CSVs in results_dir (optionally one model), with
    `length` = in-context demographic span length when recorded, else the
    standalone phrase token count."""
    paths = grid_csv_paths(results_dir)
    if model:
        paths = [p for p in paths if model_tag_from_path(p) == model]
    if not paths:
        raise SystemExit(f"No master_grid CSVs found in {results_dir}" + (f" for {model}" if model else ""))
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if "demo_span_tokens" not in df.columns:
        df["demo_span_tokens"] = np.nan
    df["length"] = df["demo_span_tokens"].where(df["demo_span_tokens"] > 0, df["phrase_tokens"])
    df["is_placebo"] = df["is_placebo"].astype(str).str.lower() == "true"
    return _fill_matched_placebo(df)


def load_perhead(npz_path, grid=None, segment=None):
    """Returns (attn, index) where attn is float32 (rows, layers, heads, 4) --
    or (rows, layers, heads) if `segment` is given -- and index is a DataFrame
    aligned to attn rows. If `grid` (that model's CSV rows) is passed,
    prob/delta/length/placebo columns are joined onto the index."""
    z = np.load(npz_path, allow_pickle=False)
    attn = z["attention"]
    attn = (attn if segment is None else attn[..., segment]).astype(np.float32)
    index = pd.DataFrame({"item_id": z["item_id"], "variant": z["variant"].astype(str),
                          "condition": z["condition"].astype(str)})
    for col in ("demo_span_tokens", "instr_span_tokens", "text_span_tokens", "seq_len"):
        if col in z.files:
            index[col] = z[col]
    if grid is not None:
        keep = ["item_id", "variant", "condition", "prob_yes", "baseline_prob_yes",
                "delta_prob_yes", "phrase_tokens", "length", "is_placebo", "axis",
                "matched_placebo", "entropy"]
        g = grid[[c for c in keep if c in grid.columns]].drop_duplicates(["item_id", "variant", "condition"])
        dup = [c for c in g.columns if c in index.columns and c not in ("item_id", "variant", "condition")]
        index = index.merge(g.drop(columns=dup), on=["item_id", "variant", "condition"], how="left")
        if "demo_span_tokens" in index.columns:
            index["length"] = index["demo_span_tokens"].where(index["demo_span_tokens"] > 0, index["length"])
    return attn, index


def bh_fdr(p):
    """Benjamini-Hochberg q-values; NaNs pass through."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def spearman_columns(x, Y):
    """Spearman rho and two-sided p between vector x (n,) and every column of
    Y (n, k), vectorized. Rows with NaN in x are dropped; Y must be NaN-free."""
    x = np.asarray(x, dtype=float)
    keep = ~np.isnan(x)
    x, Y = x[keep], np.asarray(Y, dtype=float)[keep]
    n = len(x)
    if n < 5:
        return np.full(Y.shape[1], np.nan), np.full(Y.shape[1], np.nan), n
    rx = stats.rankdata(x)
    RY = stats.rankdata(Y, axis=0)
    rx = rx - rx.mean()
    RY = RY - RY.mean(axis=0)
    denom = np.sqrt((rx ** 2).sum() * (RY ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (rx @ RY) / denom
        t = rho * np.sqrt((n - 2) / np.clip(1 - rho ** 2, 1e-12, None))
    p = 2 * stats.t.sf(np.abs(t), df=n - 2)
    return rho, p, n


def paired_columns(D):
    """One-sample tests on paired differences D (n, k), column-wise: mean,
    Cohen's dz, Wilcoxon signed-rank p. Columns that are all zero get p=NaN."""
    D = np.asarray(D, dtype=float)
    mean = D.mean(axis=0)
    sd = D.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dz = mean / sd
    p = np.full(D.shape[1], np.nan)
    nonzero = (D != 0).any(axis=0)
    if D.shape[0] >= 5 and nonzero.any():
        p[nonzero] = stats.wilcoxon(D[:, nonzero], axis=0, zero_method="wilcox").pvalue
    return mean, dz, p


def cluster_bootstrap_mean(values_by_item, n_boot=2000, seed=0):
    """95% CI of a mean over items, resampling items (the unit that is shared
    across conditions). values_by_item: 1-D array, one value per item."""
    v = np.asarray(values_by_item, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
