#!/usr/bin/env python3
"""Redraw the standalone Figure 5C panel from frozen public CSV tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "assets/manuscript_figures/full/fig5"
DEFAULT_OUTPUT = ROOT / "figures/figure_subplot_check/fig5"

REF_GRAY = "#b8b8b8"
CAND_GRAY = "#777777"
SEL_RED = "#d62728"
COLORS = {
    "ref": REF_GRAY,
    "MUT3": CAND_GRAY,
    "MUT6": CAND_GRAY,
    "MUT15": CAND_GRAY,
    "MUT17": SEL_RED,
}
ORDER = ["ref", "MUT3", "MUT6", "MUT15", "MUT17"]
MARKERS = {"ref": "o", "MUT3": "s", "MUT6": "^", "MUT15": "D", "MUT17": "*"}
LINESTYLES = {"ref": "--", "MUT3": ":", "MUT6": "--", "MUT15": "--", "MUT17": "-"}


def load_tables(data_dir: Path) -> tuple[pd.DataFrame, ...]:
    expression_raw = pd.read_csv(data_dir / "F5C_expression_raw.csv")
    expression_summary = pd.read_csv(data_dir / "F5C_expression_summary.csv")
    stability_summary = pd.read_csv(data_dir / "F5C_stability_summary.csv")
    stability_fit = pd.read_csv(data_dir / "F5C_stability_fit.csv")

    for frame, column in (
        (expression_raw, "Variant"),
        (expression_summary, "variant"),
        (stability_summary, "sample"),
        (stability_fit, "sample"),
    ):
        observed = set(frame[column].astype(str))
        missing = set(ORDER) - observed
        if missing:
            raise ValueError(f"{column} is missing displayed variants: {sorted(missing)}")
        if "MA-hek-3" in observed or "MAhek3" in observed:
            raise ValueError("industrial reference must use the public display name MUT3")

    return expression_raw, expression_summary, stability_summary, stability_fit


def exponential_decay(t: np.ndarray, k: float) -> np.ndarray:
    return 100.0 * np.exp(-k * t)


def render(data_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    expression_raw, expression_summary, stability_summary, stability_fit = load_tables(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = expression_raw.pivot(index="Variant", columns="replicate", values="protein_concentration_ng_ml")
    summary = expression_summary.set_index("variant")
    fit = stability_fit.set_index("sample")

    plt.rcParams.update({"font.size": 12})
    figure = plt.figure(figsize=(12, 14))
    grid = gridspec.GridSpec(2, 1, figure=figure, hspace=0.55, height_ratios=[1, 1])
    axis_bar = figure.add_subplot(grid[0])
    axis_stability = figure.add_subplot(grid[1])

    x = np.arange(len(ORDER))
    for index, variant in enumerate(ORDER):
        values = raw.loc[variant].dropna().to_numpy(dtype=float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        selected = variant == "MUT17"
        axis_bar.bar(
            x[index], mean, width=0.55, color=COLORS[variant], alpha=0.95,
            edgecolor="black", linewidth=2.6 if selected else 0.8,
            zorder=3 if selected else 2,
        )
        axis_bar.errorbar(x[index], mean, yerr=sd, fmt="none", color="black", capsize=5, linewidth=1.2)
        if variant != "ref":
            label = str(summary.loc[variant, "significance"])
            axis_bar.text(
                x[index], mean + sd + 0.03, label, ha="center", va="bottom",
                fontsize=12 if label != "ns" else 10, fontweight="bold",
            )

    axis_bar.set_ylim(0, 1.5)
    axis_bar.set_xlim(-0.6, len(ORDER) - 0.4)
    axis_bar.set_xticks(x)
    axis_bar.set_xticklabels(ORDER, fontsize=12)
    axis_bar.set_ylabel("Protein Conc. (ng/ml)", fontsize=12)
    axis_bar.set_title("Dual-objective NGF lead candidates", fontsize=13, fontweight="bold")
    axis_bar.spines[["top", "right"]].set_visible(False)
    axis_bar.legend(
        handles=[
            mpatches.Patch(facecolor=REF_GRAY, edgecolor="black", label="Reference / baseline"),
            mpatches.Patch(facecolor=CAND_GRAY, edgecolor="black", label="Other candidates (MUT3/6/15)"),
            mpatches.Patch(facecolor=SEL_RED, edgecolor="black", linewidth=2.6,
                           label="Selected for in vivo study (MUT17)"),
        ],
        fontsize=9.5, loc="upper left", framealpha=0.9,
    )

    for sample in ORDER:
        points = stability_summary[stability_summary["sample"] == sample].sort_values("time_hours")
        k = float(fit.loc[sample, "k_per_hour"])
        half_life = float(fit.loc[sample, "half_life_hours"])
        color = COLORS[sample]
        axis_stability.errorbar(
            points["time_hours"], points["mean_percent"], yerr=points["sd_percent"],
            fmt=MARKERS[sample], color=color,
            label=f"{sample}  ($t_{{1/2}}$={half_life:.1f} h)", capsize=5,
            markersize=13 if sample == "MUT17" else 7, alpha=0.9,
            zorder=4 if sample == "MUT17" else 3,
        )
        t_curve = np.linspace(0, 9, 200)
        axis_stability.plot(
            t_curve, exponential_decay(t_curve, k), color=color,
            linestyle=LINESTYLES[sample], linewidth=2.5 if sample == "MUT17" else 1.6,
            alpha=1.0 if sample == "MUT17" else 0.75,
        )

    axis_stability.set_title("mRNA Stability", fontsize=13, fontweight="bold")
    axis_stability.set_xlabel("Time (hours)", fontsize=12)
    axis_stability.set_ylabel("mRNA Remaining (%)", fontsize=12)
    axis_stability.set_ylim(15, 115)
    axis_stability.legend(fontsize=11, loc="upper right")
    axis_stability.grid(True, linestyle="--", alpha=0.3)
    axis_stability.spines[["top", "right"]].set_visible(False)
    axis_stability.text(
        0.02, 0.04,
        "Fit: $y = 100 \\cdot e^{-kt}$   |   $t_{1/2} = \\ln2 / k$   |   Error bars: SD,  $n = 3$",
        transform=axis_stability.transAxes, ha="left", va="bottom",
        fontsize=8.5, color="#555555", style="italic",
    )

    for axis, label in ((axis_bar, "A"), (axis_stability, "B")):
        axis.text(-0.08, 1.04, label, transform=axis.transAxes,
                  fontsize=15, fontweight="bold", va="top")

    png = output_dir / "NGF_combined.png"
    pdf = output_dir / "NGF_combined.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    png, pdf = render(args.data_dir, args.output_dir)
    print(f"saved {png}")
    print(f"saved {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
