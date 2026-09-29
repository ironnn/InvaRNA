#!/usr/bin/env python3
"""Reproduce the computational Fig. 4A summary and Fig. 4B dORF ranking table."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from invarna.interpretability.dorf_scoring import rank_dorf_candidates
from invarna.interpretability.feature_shuffle import iqr_filter_element_scores, summarize_element_scores


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets" / "manuscript_figures" / "fig4_compact"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "fig4")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.concat([
        pd.read_csv(SOURCE / "_f4a_local_s0.csv"),
        pd.read_csv(SOURCE / "_f4a_local_s1.csv"),
    ], ignore_index=True)
    scores["Expert"] = scores["Expert"].replace("Single Base", "Single-base")
    conditions = ["uORF", "RBP (5′UTR)", "RBP (CDS)", "RBP (3′UTR)", "miRNA"]
    experts = ["Single-base", "Codon", "9-mer", "Ribosome"]
    summary = summarize_element_scores(iqr_filter_element_scores(scores), conditions, experts)
    summary.to_csv(args.output_dir / "fig4a_statistics_iqr.csv", index=False)

    values = summary.pivot(index="Condition", columns="Expert", values="Mean_Z").loc[conditions, experts]
    significance = summary.pivot(index="Condition", columns="Expert", values="Significance").loc[conditions, experts]
    limit = min(float(np.nanmax(np.abs(values.to_numpy()))), 2.0)
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for row, condition in enumerate(conditions):
        for column, expert in enumerate(experts):
            value = values.loc[condition, expert]
            suffix = significance.loc[condition, expert]
            label = f"{value:.2f}" if suffix != "ns" else f"{value:.2f}\nns"
            axis.text(column, row, label, ha="center", va="center",
                      color="white" if abs(value) > limit * 0.5 else "black")
    axis.set_xticks(range(len(experts)), experts)
    axis.set_yticks(range(len(conditions)), conditions)
    fig.colorbar(image, ax=axis, label="Mean Z-Score", shrink=0.7)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig4a_element_routing.png", dpi=150)
    plt.close(fig)

    genome = pd.read_csv(SOURCE / "dorf_zscore_genome_wide_InvaRNA0412.csv")
    rank_dorf_candidates(genome).to_csv(args.output_dir / "fig4b_dorf_ribosome_rank.csv", index=False)
    print(f"Saved Fig. 4 computational outputs under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
