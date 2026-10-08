"""
Phase 2 behavioural analyses on the v2 grid (CPU only, both models).

Primary scale: change in log-odds of "yes" vs the item's own baseline
(delta_logodds_yes). It does not saturate at 0/1, so shifts are comparable
across items with different baselines. P(yes) shifts are reported alongside.

Design facts the tests rely on: every model saw the same 500 items under all
22 conditions x 10 variants (fully crossed, balanced). Within a variant, an
inserted phrase is identical on every item, so a test with items as the unit
mostly answers "do these tokens differ from those tokens" and comes out
significant for almost anything. Claims about identity CONTENT therefore use
the phrase as the unit:

  - per demographic: its mean shift against the spread of the 8 placebos'
    mean shifts in the same variant (prediction interval for one new
    content-free phrase, t with df = n_placebos - 1)
  - per axis / all demographics: two-sample t-test on condition means
    (demographics vs placebos). In a balanced crossed design this is exactly
    the test of the fixed effect in the mixed model
    shift ~ is_demographic + (1|item) + (1|phrase), because item effects are
    shared by every phrase and cancel from the phrase means.

Item-level numbers (bootstrap CIs, Wilcoxon) are kept as descriptive
precision of each phrase's effect, not as evidence about identity content.

Outputs: results/v2/analysis/behaviour_*.csv
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

RES = os.environ.get("RESULTS_DIR", "results/v2/master_grid")
OUT = os.environ.get("OUT_DIR", "results/v2/analysis")
MODELS = ["Llama-3-1-8B-Instruct", "Qwen2-5-7B-Instruct"]
VARIANTS = ["prefix", "reworded", "imagine", "consider", "infix", "suffix",
            "embedded", "target", "context", "context_long"]
PARAPHRASE = ["prefix", "reworded", "imagine", "consider"]
D3_RATINGS_CSV = "../dataset/d3-ratings.csv"
D3_RATERS_CSV = "../dataset/d3-raters.csv"
MIN_GROUP_RATERS = 4
GROUP_OF = {
    "region_western_europe": ("Region", "Western Europe"), "region_north_america": ("Region", "North America"),
    "region_latin_america": ("Region", "Latin America"), "region_arab_culture": ("Region", "Arab Culture"),
    "region_indian_cultural_sphere": ("Region", "Indian Cultural Sphere"),
    "region_sinosphere": ("Region", "Sinosphere"), "region_sub_saharan_africa": ("Region", "Sub Saharan Africa"),
    "region_oceania": ("Region", "Oceania"), "gender_man": ("Gender", "Man"), "gender_woman": ("Gender", "Woman"),
}
rng = np.random.default_rng(0)


def boot_ci(v, n_boot=2000):
    v = np.asarray(v, dtype=float)
    b = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def load():
    frames = []
    for m in MODELS:
        d = pd.read_csv(f"{RES}/master_grid_{m}.csv.gz")
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["is_placebo"] = d["is_placebo"].astype(str).str.lower() == "true"
    d["margin"] = (2 * d.prob_yes - 1).abs()
    d["baseline_margin"] = (2 * d.baseline_prob_yes - 1).abs()
    d["d_margin"] = d.margin - d.baseline_margin
    return d


def condition_table(g):
    rows = []
    for (m, v, c), x in g.groupby(["model", "variant", "condition"]):
        lo, hi = boot_ci(x.delta_logodds_yes)
        rows.append({"model": m, "variant": v, "condition": c, "axis": x.axis.iloc[0],
                     "is_placebo": x.is_placebo.iloc[0], "n": len(x),
                     "mean_dlo": x.delta_logodds_yes.mean(), "dlo_ci_lo": lo, "dlo_ci_hi": hi,
                     "mean_dp": x.delta_prob_yes.mean(), "frac_up": (x.delta_logodds_yes > 0).mean(),
                     "flip_rate": x.flipped_vs_baseline.mean(), "mean_d_margin": x.d_margin.mean(),
                     "mean_prob": x.prob_yes.mean(), "pct_demographic_norm": x.pct_demographic_norm.mean(),
                     "demo_span_tokens": x.demo_span_tokens.iloc[0]})
    return pd.DataFrame(rows)


def identity_specific(g, ct):
    """Per (model, variant, demographic): item-level contrast against its
    exact-length placebos, plus the phrase-level placement in the placebo
    distribution of that variant."""
    rows = []
    for (m, v), x in g.groupby(["model", "variant"]):
        w = x.pivot(index="item_id", columns="condition", values="delta_logodds_yes")
        cm = ct[(ct.model == m) & (ct.variant == v)].set_index("condition")
        plc = cm[cm.is_placebo]
        mp, sp, n_p = plc.mean_dlo.mean(), plc.mean_dlo.std(ddof=1), len(plc)
        plc_no_tea = plc.drop(index="placebo_tea", errors="ignore")
        # length-adjusted null: content-free phrases shift less the longer they are
        # (negative length slope in every variant), so fit that line through all
        # placebos and ask where a demographic falls relative to a NEW content-free
        # phrase of its own length (prediction interval, df = n_placebos - 2). Uses
        # all 8 placebos instead of the 2 exact-length partners.
        L = plc.demo_span_tokens.values.astype(float)
        X = np.column_stack([np.ones(n_p), L])
        coef, *_ = np.linalg.lstsq(X, plc.mean_dlo.values, rcond=None)
        res_sd = np.sqrt(((plc.mean_dlo.values - X @ coef) ** 2).sum() / (n_p - 2))
        sxx = ((L - L.mean()) ** 2).sum()
        for d, r in cm[~cm.is_placebo].iterrows():
            pred = coef[0] + coef[1] * r.demo_span_tokens
            se = res_sd * np.sqrt(1 + 1 / n_p + (r.demo_span_tokens - L.mean()) ** 2 / sxx)
            t_len = (r.mean_dlo - pred) / se
            partners = x.loc[x.condition == d, "matched_placebos"].iloc[0].split("|")
            spec = w[d] - w[partners].mean(axis=1)
            lo, hi = boot_ci(spec)
            t = (r.mean_dlo - mp) / (sp * np.sqrt(1 + 1 / n_p))
            t2 = (r.mean_dlo - plc_no_tea.mean_dlo.mean()) / (plc_no_tea.mean_dlo.std(ddof=1)
                                                               * np.sqrt(1 + 1 / len(plc_no_tea)))
            rows.append({"model": m, "variant": v, "condition": d, "axis": r.axis,
                         "matched_placebos": "|".join(partners),
                         "mean_dlo": r.mean_dlo, "matched_placebo_dlo": w[partners].mean(axis=1).mean(),
                         "identity_specific_dlo": spec.mean(), "spec_ci_lo": lo, "spec_ci_hi": hi,
                         "frac_items_spec_pos": (spec > 0).mean(),
                         "p_item_wilcoxon": stats.wilcoxon(spec)[1],
                         "placebo_mean": mp, "placebo_sd": sp, "n_placebos": n_p,
                         "t_vs_placebo_null": t, "p_phrase": 2 * stats.t.sf(abs(t), df=n_p - 1),
                         "t_vs_placebo_null_no_tea": t2,
                         "p_phrase_no_tea": 2 * stats.t.sf(abs(t2), df=len(plc_no_tea) - 1),
                         "rank_among_placebos": int((plc.mean_dlo < r.mean_dlo).sum()),
                         "placebo_length_slope": coef[1], "length_null_pred": pred,
                         "vs_length_null_dlo": r.mean_dlo - pred, "t_vs_length_null": t_len,
                         "p_length_null": 2 * stats.t.sf(abs(t_len), df=n_p - 2)})
    return pd.DataFrame(rows)


def axis_tests(g, ct):
    """Demographics vs placebos on condition means (= mixed-model fixed effect
    in this balanced design), overall and per axis, plus variance components
    from the two-way (item x condition) decomposition."""
    rows = []
    for (m, v), x in g.groupby(["model", "variant"]):
        cm = ct[(ct.model == m) & (ct.variant == v)]
        p = cm[cm.is_placebo].mean_dlo
        w = x.pivot(index="item_id", columns="condition", values="delta_logodds_yes").values
        n_i, n_c = w.shape
        gm = w.mean()
        ms_item = n_c * ((w.mean(1) - gm) ** 2).sum() / (n_i - 1)
        ms_cond = n_i * ((w.mean(0) - gm) ** 2).sum() / (n_c - 1)
        resid = w - w.mean(1, keepdims=True) - w.mean(0, keepdims=True) + gm
        ms_res = (resid ** 2).sum() / ((n_i - 1) * (n_c - 1))
        vc = {"var_item": max((ms_item - ms_res) / n_c, 0), "var_phrase": max((ms_cond - ms_res) / n_i, 0),
              "var_residual": ms_res}
        for axis in ["all", "region", "gender", "religion", "identity"]:
            dm = cm[~cm.is_placebo] if axis == "all" else cm[cm.axis == axis]
            t, pv = stats.ttest_ind(dm.mean_dlo, p, equal_var=(axis == "all"))
            # same comparison with span length as a covariate (ANCOVA on condition
            # means): placebos shift less the longer they are, and demographics differ
            # in length from the placebo set, so the raw difference mixes both
            sub = pd.concat([dm, cm[cm.is_placebo]])
            X = np.column_stack([np.ones(len(sub)), (~sub.is_placebo).astype(float), sub.demo_span_tokens])
            beta, *_ = np.linalg.lstsq(X, sub.mean_dlo.values, rcond=None)
            dof = len(sub) - 3
            s2 = ((sub.mean_dlo.values - X @ beta) ** 2).sum() / dof
            se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
            rows.append({"model": m, "variant": v, "axis": axis, "n_demographics": len(dm), "n_placebos": len(p),
                         "demographic_mean_dlo": dm.mean_dlo.mean(), "placebo_mean_dlo": p.mean(),
                         "diff": dm.mean_dlo.mean() - p.mean(), "t": t, "p_phrase_level": pv,
                         "diff_length_adj": beta[1], "p_length_adj": 2 * stats.t.sf(abs(beta[1] / se), df=dof),
                         **vc})
    return pd.DataFrame(rows)


def paraphrase_consistency(ct, spec):
    """Do the 4 identity-sentence templates agree? Spearman over the 22
    condition means and over the 14 identity-specific effects, per template pair."""
    rows = []
    for m in MODELS:
        a = ct[(ct.model == m) & ct.variant.isin(PARAPHRASE)].pivot(index="condition", columns="variant",
                                                                     values="mean_dlo")
        b = spec[(spec.model == m) & spec.variant.isin(PARAPHRASE)].pivot(
            index="condition", columns="variant", values="identity_specific_dlo")
        for i, v1 in enumerate(PARAPHRASE):
            for v2 in PARAPHRASE[i + 1:]:
                rows.append({"model": m, "template_1": v1, "template_2": v2,
                             "rho_condition_means": stats.spearmanr(a[v1], a[v2])[0],
                             "rho_identity_specific": stats.spearmanr(b[v1], b[v2])[0]})
    return pd.DataFrame(rows)


def pooled_paraphrase(g):
    """Conditions averaged over the 4 templates (prefix position): one mean per
    condition, then demographics vs placebos at the phrase level."""
    x = g[g.variant.isin(PARAPHRASE)]
    cm = x.groupby(["model", "condition", "axis", "is_placebo"]).delta_logodds_yes.mean().reset_index()
    rows = []
    for m, c in cm.groupby("model"):
        p = c[c.is_placebo].delta_logodds_yes
        for axis in ["all", "region", "gender", "religion", "identity"]:
            dm = c[~c.is_placebo] if axis == "all" else c[c.axis == axis]
            t, pv = stats.ttest_ind(dm.delta_logodds_yes, p, equal_var=(axis == "all"))
            rows.append({"model": m, "axis": axis, "demographic_mean_dlo": dm.delta_logodds_yes.mean(),
                         "placebo_mean_dlo": p.mean(), "diff": dm.delta_logodds_yes.mean() - p.mean(),
                         "t": t, "p_phrase_level": pv})
    return pd.DataFrame(rows), cm


def moderators(g):
    """Item-level: identity-specific effect averaged over demographics and the
    4 templates vs item covariates; also the general framing shift (all
    conditions) vs baseline log-odds."""
    rows = []
    x = g[g.variant.isin(PARAPHRASE)]
    for m, xm in x.groupby("model"):
        item = xm.groupby(["item_id", "is_placebo"]).delta_logodds_yes.mean().unstack()
        meta = xm.drop_duplicates("item_id").set_index("item_id")
        spec = item[False] - item[True]
        allshift = xm.groupby("item_id").delta_logodds_yes.mean()
        for cov in ["entropy", "p_offensive_raters", "baseline_logodds_yes"]:
            rows.append({"model": m, "outcome": "identity_specific", "covariate": cov,
                         "spearman": stats.spearmanr(spec, meta.loc[spec.index, cov])[0],
                         "p": stats.spearmanr(spec, meta.loc[spec.index, cov])[1]})
            rows.append({"model": m, "outcome": "any_phrase_shift", "covariate": cov,
                         "spearman": stats.spearmanr(allshift, meta.loc[allshift.index, cov])[0],
                         "p": stats.spearmanr(allshift, meta.loc[allshift.index, cov])[1]})
        for cat, ids in meta.groupby("category").groups.items():
            rows.append({"model": m, "outcome": "identity_specific", "covariate": f"category={cat}",
                         "mean": spec.loc[ids].mean(), "n": len(ids)})
            rows.append({"model": m, "outcome": "any_phrase_shift", "covariate": f"category={cat}",
                         "mean": allshift.loc[ids].mean(), "n": len(ids)})
        groups = [spec.loc[ids] for ids in meta.groupby("category").groups.values()]
        rows.append({"model": m, "outcome": "identity_specific", "covariate": "category (Kruskal)",
                     "p": stats.kruskal(*groups)[1]})
    return pd.DataFrame(rows)


def group_rates(item_ids):
    """Per item and rater group: fraction of that group's raters calling it offensive."""
    r = pd.read_csv(D3_RATINGS_CSV)
    r = r[r.item_id.isin(item_ids)].dropna(subset=["rating_binary"])
    raters = pd.read_csv(D3_RATERS_CSV)[["rater_id", "Gender", "Region"]]
    r = r.merge(raters, on="rater_id", how="left")
    out = {}
    for cond, (col, val) in GROUP_OF.items():
        s = r[r[col] == val].groupby("item_id").rating_binary.agg(["mean", "size"])
        out[cond] = s[s["size"] >= MIN_GROUP_RATERS]["mean"]
    return out


def group_alignment(d):
    """Absolute alignment: is the framed P(yes) closer to the framed group's
    real offensive rate than the unframed P(yes), and than the same item's
    matched placebos? Mean absolute error over items with enough raters."""
    base = d[d.condition == "baseline"].drop_duplicates(["model", "item_id"]).set_index(["model", "item_id"])
    rates = group_rates(d.item_id.unique())
    rows = []
    g = d[d.condition != "baseline"]
    for (m, v), x in g.groupby(["model", "variant"]):
        w = x.pivot(index="item_id", columns="condition", values="prob_yes")
        for cond, rate in rates.items():
            ids = rate.index.intersection(w.index)
            partners = x.loc[x.condition == cond, "matched_placebos"].iloc[0].split("|")
            pb = base.loc[m].loc[ids, "prob_yes"]
            framed, plc = w.loc[ids, cond], w.loc[ids, partners].mean(axis=1)
            rows.append({"model": m, "variant": v, "condition": cond, "n_items": len(ids),
                         "group_rate_mean": rate.loc[ids].mean(),
                         "mae_baseline": (pb - rate.loc[ids]).abs().mean(),
                         "mae_framed": (framed - rate.loc[ids]).abs().mean(),
                         "mae_matched_placebo": (plc - rate.loc[ids]).abs().mean(),
                         "rho_framed_vs_rate": stats.spearmanr(framed, rate.loc[ids])[0],
                         "rho_baseline_vs_rate": stats.spearmanr(pb, rate.loc[ids])[0]})
    return pd.DataFrame(rows)


def cross_model(ct, spec):
    rows = []
    for name, tab, col in [("condition_means", ct, "mean_dlo"), ("identity_specific", spec, "identity_specific_dlo")]:
        w = tab.pivot_table(index=["variant", "condition"], columns="model", values=col)
        rows.append({"what": name, "scope": "all variants", "n": len(w),
                     "spearman": stats.spearmanr(w[MODELS[0]], w[MODELS[1]])[0]})
        for v, x in w.groupby(level="variant"):
            rows.append({"what": name, "scope": v, "n": len(x),
                         "spearman": stats.spearmanr(x[MODELS[0]], x[MODELS[1]])[0]})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load()
    g = d[d.condition != "baseline"]
    assert g.groupby(["model", "variant", "condition"]).size().nunique() == 1, "unbalanced design"
    ct = condition_table(g)
    spec = identity_specific(g, ct)
    ax = axis_tests(g, ct)
    para = paraphrase_consistency(ct, spec)
    pooled, pooled_cm = pooled_paraphrase(g)
    mod = moderators(g)
    ga = group_alignment(d)
    cm = cross_model(ct, spec)
    for name, tab in [("conditions", ct), ("identity_specific", spec), ("axis_tests", ax),
                      ("paraphrase_consistency", para), ("paraphrase_pooled", pooled),
                      ("paraphrase_pooled_condition_means", pooled_cm), ("moderators", mod),
                      ("group_alignment", ga), ("cross_model", cm)]:
        tab.to_csv(f"{OUT}/behaviour_{name}.csv", index=False)
    print(f"wrote behaviour_*.csv to {OUT}")


if __name__ == "__main__":
    main()
