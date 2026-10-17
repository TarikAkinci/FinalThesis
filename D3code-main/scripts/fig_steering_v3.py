"""
Figure for run C: distance to the real group's average rating vs steering
strength (prefix format, held-out split B), per model. Lines: dampener heads,
text steering, random heads (same layer mix as the dampeners). Reference lines:
the unsteered prompt and a constant downward shift fitted on split A.
Colours: validated categorical slots 1-3 (dataviz reference palette); the
aqua slot is below 3:1 contrast, so every line is directly labelled.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

S = pd.read_csv("results/v3/analysis/steering_summary.csv")
T = pd.read_csv("results/v3/analysis/steering_vs_trivial.csv")
OUT = "figures/v3/steering_dose_response.png"
MODELS = {"Llama-3-1-8B-Instruct": "Llama-3.1-8B", "Qwen2-5-7B-Instruct": "Qwen2.5-7B"}
LEVELS = [0, 0.5, 1, 2, 4, 10, 100]
LABELS = ["0\n(removed)", "0.5", "1\n(none)", "2", "4", "10", "100\n(PASTA)"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = {"dampeners": ("#2a78d6", "o", "identity steering, dampener heads"),
          "text": ("#eb6834", "s", "steering toward the text"),
          "random": ("#1baf7a", "^", "random heads (same layers)")}


def curve(model, label, head_filter):
    x = S[(S.model == model) & (S.set_label == label) & (S.variant == "prefix")
          & S.axis.isin(["region", "gender"]) & head_filter(S.head_set)]
    by = x.groupby("strength").mae_group.mean()
    ref = x.mae_group_ref.mean()
    by.loc[1.0] = ref
    return by.reindex(LEVELS)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, facecolor=SURF)
    xs = np.arange(len(LEVELS))
    for ax, (model, name) in zip(axes, MODELS.items()):
        ax.set_facecolor(SURF)
        lines = {"dampeners": curve(model, "attention_identity_dampeners", lambda h: h.isin(["selected", "none"])),
                 "text": curve(model, "attention_text", lambda h: h.isin(["selected", "none"])),
                 "random": curve(model, "attention_identity_dampeners",
                                 lambda h: h.str.startswith("random") | (h == "none"))}
        unsteered = lines["dampeners"].loc[1.0]
        const = T[(T.model == model) & (T.variant == "prefix") & (T.target == "group")].mae_constant_shift.iloc[0]
        ax.axhline(unsteered, color=INK2, lw=1, ls=(0, (1, 2)))
        ax.text(-0.25, unsteered + 0.012, "unsteered", va="bottom", color=INK2, fontsize=8.5)
        ax.axhline(const, color=INK2, lw=1, ls=(0, (6, 3)))
        ax.text(-0.25, const - 0.012, "constant shift (trivial fix)", va="top", color=INK2, fontsize=8.5)
        # direct labels at the line ends, nudged apart so they never overlap
        ends = sorted(((y.values[-1], k) for k, y in lines.items()), reverse=True)
        placed = []
        for val, k in ends:
            pos = val if not placed else min(val, placed[-1] - 0.045)
            placed.append(pos)
            c, mk, lab = SERIES[k]
            y = lines[k]
            ax.plot(xs, y.values, color=c, lw=2, marker=mk, ms=8, markeredgecolor=SURF, markeredgewidth=2)
            ax.annotate(lab, (xs[-1], val), xytext=(xs[-1] + 0.25, pos), textcoords="data",
                        va="center", fontsize=9, color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels(LABELS, fontsize=9, color=INK2)
        ax.set_xlim(-0.3, len(LEVELS) + 1.6)
        ax.set_title(name, loc="left", fontsize=12, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="y", colors=INK2, length=0)
        ax.set_xlabel("steering strength (multiplier on attention to the steered span)", fontsize=9, color=INK2)
    axes[0].set_ylabel("distance to the real group's average\n(mean |P(offensive) - share of raters|)", fontsize=9, color=INK2)
    handles = [plt.Line2D([], [], color=c, marker=mk, lw=2, ms=7) for c, mk, _ in SERIES.values()]
    fig.suptitle("Steering brings the answers closer to real raters, but no closer than a constant shift "
                 "(prefix format, held-out texts)", y=1.0, fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(handles, [v[2] for v in SERIES.values()], loc="upper center", bbox_to_anchor=(0.5, 0.93),
               ncol=3, frameon=False, fontsize=9)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
