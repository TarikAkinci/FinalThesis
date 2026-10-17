"""
Scores the run-C steering outputs (steer_eval.py) against the success
criterion agreed with the supervisor (10 Oct 2026): does steering bring the
model closer to the real people's average rating? CPU only.

Real average per item:
  framed prompt (region / gender)  share of THAT group's raters calling the
                                   text offensive (items with >= MIN_GROUP_RATERS)
  unframed prompt, and everything  share of ALL raters (D3CODE rating_binary)
  (religion has no rater data in D3CODE: only shifts are reported)

Per (model, method, target, head_set, strength, variant, condition):
  mae_group / mae_all       mean |P(yes) - real share| over the held-out items
  d_mae_*                   steered minus unsteered (negative = closer = success)
  shift_lo                  mean change in log-odds vs unsteered, same prompt
  identity_specific_shift   shift(condition) - mean shift(its length-matched placebos)
  place_specific_shift      shift(region) - mean shift(its place-name controls)
  false_pos_rate            P(yes) > 0.5 on texts most raters call NOT offensive
Successes are judged against the random head sets (same layer mix) and against
steering of the placebo spans, written next to each other in the output.

  python analysis_steering.py [--dir results/v3/steering] [--out results/v3/analysis]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from analysis_v2_behaviour import GROUP_OF, D3_RATINGS_CSV, D3_RATERS_CSV, MIN_GROUP_RATERS

ITEMS_CSV = "items/items_v2.csv"
V2_GRID = "results/v2/master_grid/master_grid_{model}.csv.gz"


def group_rates(item_ids):
    r = pd.read_csv(D3_RATINGS_CSV)
    r = r[r.item_id.isin(item_ids)].dropna(subset=["rating_binary"])
    r = r.merge(pd.read_csv(D3_RATERS_CSV)[["rater_id", "Gender", "Region"]], on="rater_id", how="left")
    rows = []
    for cond, (col, val) in GROUP_OF.items():
        s = r[r[col] == val].groupby("item_id").rating_binary.agg(["mean", "size"])
        s = s[s["size"] >= MIN_GROUP_RATERS]
        rows.append(pd.DataFrame({"condition": cond, "item_id": s.index, "group_rate": s["mean"].values}))
    return pd.concat(rows, ignore_index=True)


def partners(model):
    """condition -> its length-matched placebos, from the v2 grid of that model."""
    g = pd.read_csv(V2_GRID.format(model=model), usecols=["condition", "matched_placebos"]).drop_duplicates()
    g = g[g.matched_placebos.fillna("") != ""]
    return {c: p.split("|") for c, p in zip(g.condition, g.matched_placebos)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/v3/steering")
    ap.add_argument("--out", default="results/v3/analysis")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import master_grid as mg

    paths = sorted(glob.glob(f"{a.dir}/steering_*.csv.gz"))
    if not paths:
        raise SystemExit(f"no steering outputs in {a.dir}")
    d = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    items = pd.read_csv(ITEMS_CSV).set_index("item_id")
    d["all_rate"] = d.item_id.map(items.p_offensive)
    d["human_majority"] = (d.all_rate >= 0.5).astype(int)
    d = d.merge(group_rates(d.item_id.unique()), on=["condition", "item_id"], how="left")

    # unsteered reference of the same prompt: strength 1 (attention) / 0 (actadd)
    if "set_label" not in d.columns:
        d["set_label"] = ""
    key = ["model", "set_label", "method", "target", "variant", "item_id", "condition"]
    ref = d[((d.method == "attention") & (d.strength == 1)) | ((d.method == "actadd") & (d.strength == 0))]
    ref = ref.drop_duplicates(key).set_index(key)[["logodds_yes", "prob_yes"]].add_suffix("_ref")
    d = d.join(ref, on=key)
    d["shift_lo"] = d.logodds_yes - d.logodds_yes_ref
    d["abs_err_all"] = (d.prob_yes - d.all_rate).abs()
    d["abs_err_all_ref"] = (d.prob_yes_ref - d.all_rate).abs()
    d["abs_err_group"] = (d.prob_yes - d.group_rate).abs()
    d["abs_err_group_ref"] = (d.prob_yes_ref - d.group_rate).abs()
    d["false_pos"] = ((d.prob_yes > 0.5) & (d.human_majority == 0)).astype(float)

    gcols = ["model", "set_label", "method", "target", "head_set", "strength", "variant", "condition"]
    s = d.groupby(gcols).agg(
        n=("item_id", "size"), mean_prob=("prob_yes", "mean"), shift_lo=("shift_lo", "mean"),
        mae_all=("abs_err_all", "mean"), mae_all_ref=("abs_err_all_ref", "mean"),
        n_group=("group_rate", "count"), mae_group=("abs_err_group", "mean"),
        mae_group_ref=("abs_err_group_ref", "mean"), false_pos_rate=("false_pos", "mean")).reset_index()
    s["d_mae_all"] = s.mae_all - s.mae_all_ref
    s["d_mae_group"] = s.mae_group - s.mae_group_ref
    s["axis"] = s.condition.map(lambda c: "baseline" if c == "baseline" else mg.CONDITIONS[c][1])

    # identity- and place-specific shifts (contrasts at the same setting, same items)
    spec = []
    for model, x in d.groupby("model"):
        prt = partners(model)
        for setting, y in x.groupby(["set_label", "method", "target", "head_set", "strength", "variant"]):
            w = y.pivot_table(index="item_id", columns="condition", values="shift_lo")
            for c in w.columns:
                if c == "baseline" or mg.CONDITIONS[c][2]:
                    continue
                row = dict(zip(["model", "set_label", "method", "target", "head_set", "strength", "variant"],
                               (model, *setting)))
                row["condition"] = c
                p = [q for q in prt.get(c, []) if q in w.columns]
                if p:
                    row["identity_specific_shift"] = (w[c] - w[p].mean(axis=1)).mean()
                pc = [q for q in mg.PLACE_CONTROL_OF.get(c, []) if q in w.columns]
                if pc:
                    row["place_specific_shift"] = (w[c] - w[pc].mean(axis=1)).mean()
                spec.append(row)
    s = s.merge(pd.DataFrame(spec), on=gcols, how="left")

    # success vs controls: selected heads' distance change against the random sets'
    sel = s[s.head_set == "selected"]
    rnd = s[s.head_set.str.startswith("random")].groupby(
        ["model", "set_label", "method", "target", "strength", "variant", "condition"]).d_mae_group.agg(["mean", "min"])
    rnd.columns = ["d_mae_group_random_mean", "d_mae_group_random_best"]
    sel = sel.join(rnd, on=["model", "set_label", "method", "target", "strength", "variant", "condition"])
    sel["success_group"] = (sel.d_mae_group < 0) & (sel.d_mae_group < sel.d_mae_group_random_best)
    s = s.merge(sel[gcols + ["d_mae_group_random_mean", "d_mae_group_random_best", "success_group"]],
                on=gcols, how="left")
    s.to_csv(f"{a.out}/steering_summary.csv", index=False)

    pd.set_option("display.width", 250)
    show = s[(s.head_set.isin(["selected", "none"])) & s.axis.isin(["region", "gender", "baseline"])]
    print(show.pivot_table(index=["model", "set_label", "method", "target", "variant", "condition"], columns="strength",
                           values="d_mae_group").round(3).to_string())
    print(f"\nwrote {a.out}/steering_summary.csv ({len(s)} rows)")


if __name__ == "__main__":
    main()
