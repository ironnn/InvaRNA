#!/usr/bin/env python3
"""Reuse the final F5D grouped-bar layout with checked public source data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "assets/manuscript_figures/full/fig5/F5D_plot_data.csv"
OUTPUT = ROOT / "figures/figure_subplot_check/fig5/F5D_0205"


def main() -> None:
    frame = pd.read_csv(DATA)
    batches = ["100ng", "500ng"]
    labels = ["Mock", "Reference", "MUT3", "MUT6", "MUT15", "MUT17"]
    colors = {
        "Mock": "#bdc3c7", "Reference": "#999999", "MUT3": "#555555",
        "MUT6": "#c0392b", "MUT15": "#922b21", "MUT17": "#e74c3c",
    }
    x = np.arange(len(batches)) * 1.3
    width = 0.82 / len(labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    for index, label in enumerate(labels):
        values = [float(frame.loc[(frame["batch"].eq(batch)) & (frame["display_label"].eq(label)), "Relative"].iloc[0])
                  for batch in batches]
        offset = (index - (len(labels) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width=width, label=label,
                      color=colors[label], edgecolor="black", alpha=0.85)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xticks(x, ["100ng", "500ng"], fontsize=11)
    ax.set_title("Relative Expression", fontsize=14, pad=15)
    ax.set_ylabel("Relative Expression Level", fontsize=12)
    ax.set_xlabel("Experimental Batch", fontsize=12)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylim(0, float(frame["Relative"].max()) * 1.18 + 0.2)
    ax.legend(title="Label", frameon=False, ncol=3)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved {OUTPUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
