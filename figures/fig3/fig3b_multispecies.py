#!/usr/bin/env python
"""Recompute the current-manuscript Fig. 3B five-species annotations and panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "benchmarks/configs/fig3_current_paper.yaml"
DEFAULT_DATA_DIR = (
    ROOT
    / "assets/manuscript_figures/full/fig3_current_nm"
)


def load_frames(data_dir: Path, config: dict) -> dict[str, pd.DataFrame]:
    return {
        model: pd.read_csv(data_dir / filename)
        for model, filename in config["panel_b"]["source_predictions"].items()
    }


def calculate(frames: dict[str, pd.DataFrame], config: dict) -> pd.DataFrame:
    panel = config["panel_b"]
    rows = []
    for model, prediction in panel["prediction_columns"].items():
        frame = frames[model]
        for species in panel["displayed_species"]:
            subset = frame.loc[frame["display"].eq(species)].dropna(
                subset=[panel["truth_column"], prediction]
            )
            subset = subset.loc[subset[panel["truth_column"]] > 0]
            rows.append(
                {
                    "model": model,
                    "display": species,
                    "species": subset["species"].iloc[0],
                    "clade": subset["clade"].iloc[0],
                    "n": len(subset),
                    "spearman": spearmanr(
                        subset[panel["truth_column"]], subset[prediction]
                    ).statistic,
                }
            )
    return pd.DataFrame(rows)


def validate(metrics: pd.DataFrame, config: dict) -> None:
    tolerance = float(config["validation"]["metric_tolerance"])
    expected = config["panel_b"]["expected"]
    for row in metrics.itertuples(index=False):
        target = float(expected[row.model][row.display])
        if abs(float(row.spearman) - target) > tolerance:
            raise RuntimeError(
                f"Fig. 3B mismatch for {row.model}/{row.display}: "
                f"{row.spearman} != {target}"
            )


def plot(
    frames: dict[str, pd.DataFrame], metrics: pd.DataFrame, config: dict, output: Path
) -> None:
    panel = config["panel_b"]
    species = panel["displayed_species"]
    colors = {
        "Human": "#b22222",
        "Mammals": "#1f77b4",
        "Vertebrates": "#17becf",
        "Invertebrates": "#2ca02c",
    }
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})
    sns.set_style("white")
    figure, axes = plt.subplots(1, 3, figsize=(8.2, 2.7), sharex=True)
    for axis, (model, prediction) in zip(axes, panel["prediction_columns"].items()):
        frame = frames[model]
        for name in species:
            subset = frame.loc[frame["display"].eq(name)].dropna(subset=[prediction])
            if len(subset) < 30:
                continue
            color = colors[subset["clade"].iloc[0]]
            sns.kdeplot(subset[prediction], ax=axis, color=color, linestyle="--", linewidth=1.2)
        lines = []
        model_metrics = metrics.loc[metrics["model"].eq(model)].set_index("display")
        for name in species:
            lines.append(f"{name:<13s} {model_metrics.loc[name, 'spearman']:.2f}")
        axis.text(
            0.98,
            0.98,
            "\n".join(lines),
            transform=axis.transAxes,
            ha="right",
            va="top",
            family="monospace",
            fontsize=6.5,
        )
        axis.set_title(model, fontweight="bold")
        axis.set_xlabel("Log10(TE)")
        axis.set_ylabel("Density")
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/fig3b")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    frames = load_frames(args.data_dir, config)
    metrics = calculate(frames, config)
    validate(metrics, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "F3B_species_spearman_no_fly.csv", index=False)
    plot(frames, metrics, config, args.output_dir / "fig3b_cross_species.png")
    print(metrics.to_string(index=False))
    print(f"PASS: reproduced current-manuscript Fig. 3B in {args.output_dir}")


if __name__ == "__main__":
    main()
