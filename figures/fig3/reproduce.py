#!/usr/bin/env python
"""Render the manuscript Fig. 3 ablation summary from committed source values."""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=ROOT / "assets/manuscript_figures/results/fig3_ablation.csv", type=Path)
    parser.add_argument("--output", default=ROOT / "results/fig3_ablation.png", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(frame["configuration"], frame["test_r2"], color="#3f718c")
    axis.set_ylabel("Held-out human TE $R^2$")
    axis.set_ylim(0.64, 0.77)
    axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
