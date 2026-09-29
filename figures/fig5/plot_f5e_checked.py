#!/usr/bin/env python3
"""Render the 20x Fig. 5E panel from checked public summary tables."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "assets/manuscript_figures/full/fig5"
OUTPUT = ROOT / "figures/figure_subplot_check/fig5/RGC_Survival_BarChart"
ORDER = ["ONC", "ONC-EGFP", "NGF-Ref-500ng", "NGF-MUT17-100ng", "NGF-MUT17-500ng"]
COLORS = ["#8F8F8F", "#B9B9B9", "#D94848", "#E77C7C", "#B53737"]


def main() -> None:
    summary = pd.read_csv(DATA / "F5E_summary.csv")
    summary = summary.loc[summary["magnification"].eq("20x")].set_index("group").loc[ORDER]
    statistics = pd.read_csv(DATA / "F5E_statistics.csv")
    statistics = statistics.loc[statistics["magnification"].eq("20x")]
    x = np.arange(len(ORDER))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.bar(x, summary["mean_rgc_count"], yerr=summary["sd_rgc_count"], capsize=6,
           color=COLORS, edgecolor="black", linewidth=1.2, alpha=0.9, zorder=1)
    index = {name: i for i, name in enumerate(ORDER)}
    top = (summary["mean_rgc_count"] + summary["sd_rgc_count"]).max()
    level = top + 2.5
    for row in statistics.itertuples(index=False):
        lo, hi = index[row.group_1], index[row.group_2]
        y = level
        ax.plot([lo, lo, hi, hi], [y, y + 1.2, y + 1.2, y], color="black", lw=1.0)
        ax.text((lo + hi) / 2, y + 1.5, row.significance, ha="center", va="bottom",
                fontsize=10 if row.significance != "ns" else 9, fontweight="bold")
        level += 4.0
    ax.set_xticks(x, ORDER, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Number of RGCs", fontsize=12, fontweight="bold")
    ax.set_title("20x Magnification", fontsize=14, fontweight="bold", pad=10)
    ax.set_ylim(20, level + 3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {OUTPUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
