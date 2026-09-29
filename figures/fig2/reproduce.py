#!/usr/bin/env python
"""Validate the archived Figure 2 source tables and redraw the computational figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "benchmarks/configs/fig2.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hashes(source_dir: Path) -> int:
    checksum_file = source_dir / "SHA256SUMS"
    entries = 0
    for line in checksum_file.read_text().splitlines():
        if not line.strip():
            continue
        expected, filename = line.split("  ", 1)
        path = source_dir / filename
        if not path.is_file():
            raise RuntimeError(f"missing Fig. 2 source file: {filename}")
        if sha256(path) != expected:
            raise RuntimeError(f"Fig. 2 source checksum mismatch: {filename}")
        entries += 1
    return entries


def load_tables(source_dir: Path) -> dict[str, pd.DataFrame]:
    names = {
        "umap": "F2A_left_species_UMAP.csv",
        "centroids": "F2A_right_centroid_vs_divergence.csv",
        "mrl": "F2B_mean_ribosome_load.csv",
        "half_life": "F2B_mrna_half_life.csv",
        "expression": "F2B_expression_level.csv",
        "routing": "F2C_left_routing_curve.csv",
        "regions": "F2C_right_region_specialization.csv",
        "manifest": "F2C_sample_manifest_seed42.csv",
    }
    return {key: pd.read_csv(source_dir / filename) for key, filename in names.items()}


def validate(tables: dict[str, pd.DataFrame], config: dict, source_dir: Path) -> dict:
    for name, expected in config["expected_rows"].items():
        observed = len(tables[name])
        if observed != expected:
            raise RuntimeError(f"Fig. 2 {name}: expected {expected} rows, found {observed}")
        if tables[name].isna().any().any():
            raise RuntimeError(f"Fig. 2 {name}: missing values are not allowed")

    counts = tables["umap"]["display_species"].value_counts().to_dict()
    expected_species = {
        species: config["f2a"]["samples_per_species"] for species in config["f2a"]["species"]
    }
    if counts != expected_species:
        raise RuntimeError(f"Fig. 2A species counts differ: {counts}")

    rho, pvalue = spearmanr(
        tables["centroids"]["timetree_divergence_time_mya"],
        tables["centroids"]["embedding_centroid_euclidean_distance"],
    )
    metric_tolerance = float(config["validation"]["metric_tolerance"])
    if abs(rho - float(config["f2a"]["expected_spearman_rho"])) > metric_tolerance:
        raise RuntimeError(f"Fig. 2A Spearman mismatch: {rho}")
    if abs(pvalue - float(config["f2a"]["expected_spearman_pvalue"])) > metric_tolerance:
        raise RuntimeError(f"Fig. 2A p-value mismatch: {pvalue}")

    b_tables = {
        "mean_ribosome_load": tables["mrl"],
        "mrna_half_life": tables["half_life"],
        "expression_level": tables["expression"],
    }
    invarna_values = {}
    for task, frame in b_tables.items():
        if not frame["test_r2"].is_monotonic_decreasing:
            raise RuntimeError(f"Fig. 2B {task} is not sorted by test R2")
        row = frame.loc[frame["Model"].eq("InvaRNA backbone")]
        if len(row) != 1:
            raise RuntimeError(f"Fig. 2B {task} lacks one InvaRNA backbone row")
        observed = float(row.iloc[0]["test_r2"])
        expected = float(config["f2b"]["invarna_test_r2"][task])
        if abs(observed - expected) > metric_tolerance:
            raise RuntimeError(f"Fig. 2B {task} R2 mismatch: {observed}")
        invarna_values[task] = observed
    if b_tables["mean_ribosome_load"]["model"].eq("codonbert").any():
        raise RuntimeError("Fig. 2B MRL archive unexpectedly contains excluded CodonBERT")

    routing = tables["routing"]
    raw_columns = [f"{expert}_raw_mean_probability" for expert in config["f2c"]["experts"]]
    valid = routing["non_padding_sequence_count"].gt(0)
    probability_error = float(
        np.max(np.abs(routing.loc[valid, raw_columns].sum(axis=1).to_numpy() - 1.0))
    )
    if probability_error > float(config["validation"]["probability_tolerance"]):
        raise RuntimeError(f"Fig. 2C per-position probabilities do not sum to one: {probability_error}")
    rolling_error = 0.0
    for expert in config["f2c"]["experts"]:
        recalculated = routing[f"{expert}_raw_mean_probability"].rolling(
            window=config["f2c"]["rolling_window"], center=True, min_periods=1
        ).mean()
        rolling_error = max(
            rolling_error,
            float(
                np.max(
                    np.abs(
                        recalculated - routing[f"{expert}_rolling80_mean_probability"]
                    )
                )
            ),
        )
    if rolling_error > float(config["validation"]["rolling_tolerance"]):
        raise RuntimeError(f"Fig. 2C rolling curves differ: {rolling_error}")
    region_sums = tables["regions"].groupby("region")["mean_expert_probability"].sum()
    region_error = float(np.max(np.abs(region_sums.to_numpy() - 1.0)))
    if region_error > float(config["validation"]["probability_tolerance"]):
        raise RuntimeError(f"Fig. 2C regional probabilities do not sum to one: {region_error}")

    manifest = tables["manifest"].sort_values("batch_order")
    if manifest["batch_order"].tolist() != list(range(1, config["f2c"]["batch_size"] + 1)):
        raise RuntimeError("Fig. 2C manifest batch order is invalid")
    fixture_metadata = json.loads((source_dir / "F2C_fixture_metadata.json").read_text())
    if fixture_metadata["batch_size"] != config["f2c"]["batch_size"]:
        raise RuntimeError("Fig. 2C fixture metadata batch size differs from config")

    return {
        "f2a_centroid_spearman_pvalue": float(pvalue),
        "f2a_centroid_spearman_rho": float(rho),
        "f2a_species_counts": counts,
        "f2b_invarna_test_r2": invarna_values,
        "f2c_position_probability_max_abs_error": probability_error,
        "f2c_region_probability_max_abs_error": region_error,
        "f2c_rolling_max_abs_error": rolling_error,
        "f2c_transcripts": len(manifest),
    }


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.labelweight": "bold",
            "axes.linewidth": 1.2,
        }
    )


def render(tables: dict[str, pd.DataFrame], output_dir: Path, config: dict) -> None:
    set_style()
    umap = tables["umap"]
    centroid = tables["centroids"]
    routing = tables["routing"]
    regions = tables["regions"]
    experts = config["f2c"]["experts"]
    drawing_order = config["f2a"]["drawing_order"]
    palette = config["f2a"]["palette"]

    figure = plt.figure(figsize=(22, 13), dpi=300)
    outer = gridspec.GridSpec(2, 1, figure=figure, hspace=0.45, height_ratios=[1, 0.85])
    top = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[0], wspace=0.374, width_ratios=[1.0, 0.85, 2.5]
    )

    axis_umap = figure.add_subplot(top[0])
    for species in drawing_order:
        subset = umap.loc[umap["display_species"].eq(species)]
        sns.kdeplot(
            x=subset["umap_dimension_1"],
            y=subset["umap_dimension_2"],
            fill=True,
            thresh=0.05,
            levels=5,
            alpha=0.15,
            color=palette[species],
            ax=axis_umap,
            zorder=2,
        )
    for index, species in enumerate(drawing_order):
        subset = umap.loc[umap["display_species"].eq(species)]
        axis_umap.scatter(
            subset["umap_dimension_1"],
            subset["umap_dimension_2"],
            color=palette[species],
            s=7,
            alpha=0.6,
            label=species,
            linewidth=0,
            zorder=3 + index,
        )
    axis_umap.spines[["top", "right"]].set_visible(False)
    axis_umap.set_xlabel("UMAP dimension 1", fontsize=12)
    axis_umap.set_ylabel("UMAP dimension 2", fontsize=12)
    legend = axis_umap.legend(
        title="Species", loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9
    )
    for handle in legend.legend_handles:
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([45])
    axis_umap.text(-0.13, 1.06, "A", transform=axis_umap.transAxes, fontsize=18, fontweight="bold")

    axis_centroid = figure.add_subplot(top[1])
    axis_centroid.scatter(
        centroid["timetree_divergence_time_mya"],
        centroid["embedding_centroid_euclidean_distance"],
        c=centroid["plot_color"],
        s=45,
        zorder=3,
        edgecolors="black",
        linewidths=0.4,
    )
    slope, intercept = np.polyfit(
        centroid["timetree_divergence_time_mya"],
        centroid["embedding_centroid_euclidean_distance"],
        1,
    )
    xline = np.linspace(
        centroid["timetree_divergence_time_mya"].min(),
        centroid["timetree_divergence_time_mya"].max(),
        100,
    )
    axis_centroid.plot(xline, slope * xline + intercept, color="#E64B35", linewidth=1.5, linestyle="--")
    rho, pvalue = spearmanr(
        centroid["timetree_divergence_time_mya"], centroid["embedding_centroid_euclidean_distance"]
    )
    axis_centroid.set_xlabel("TimeTree divergence time (Mya)", fontsize=11)
    axis_centroid.set_ylabel("Embedding centroid distance", fontsize=11)
    axis_centroid.set_title("Embedding vs\nTimeTree divergence", fontsize=10, color="#555555")
    ptext = "p < 0.001" if pvalue < 0.001 else f"p = {pvalue:.3f}"
    axis_centroid.text(
        0.05,
        0.95,
        f"Spearman r = {rho:.2f}\n{ptext}\nn = {len(centroid)} species pairs",
        transform=axis_centroid.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.8),
    )
    axis_centroid.spines[["top", "right"]].set_visible(False)
    legend_items = [
        Line2D([0], [0], marker="o", color="white", markerfacecolor="#4DBBD5", markersize=7, label="Vertebrate pairs"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor="#F39B7F", markersize=7, label="Vertebrate–non-vertebrate pairs"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor="#8491B4", markersize=7, label="Non-vertebrate pairs"),
    ]
    axis_centroid.legend(handles=legend_items, frameon=False, fontsize=7.5, loc="lower right")

    benchmark_grid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=top[2], wspace=0.3)
    benchmark_panels = [
        (tables["mrl"], "Mean ribosome load"),
        (tables["half_life"], "mRNA half-life"),
        (tables["expression"], "Expression level"),
    ]
    for index, (frame, title) in enumerate(benchmark_panels):
        axis = figure.add_subplot(benchmark_grid[index])
        is_invarna = frame["is_invarna_backbone"].astype(str).str.lower().eq("true")
        colors = np.where(is_invarna, "#E64B35", "#808183")
        axis.bar(frame["Model"], frame["test_r2"], color=colors, edgecolor="black", linewidth=0.8, width=0.7)
        axis.set_title(title, fontsize=11, fontweight="bold", pad=10)
        if index == 0:
            axis.set_ylabel("Test $R^2$", fontsize=10, fontweight="bold")
            axis.text(-0.28, 1.06, "B", transform=axis.transAxes, fontsize=18, fontweight="bold")
        axis.set_ylim(0, 0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", labelrotation=45, labelsize=8)
        for tick in axis.get_xticklabels():
            tick.set_ha("right")
        invarna_index = frame.index[frame["Model"].eq("InvaRNA backbone")][0]
        value = float(frame.loc[invarna_index, "test_r2"])
        axis.text(invarna_index, value + 0.014, f"{value:.2f}", ha="center", fontsize=9, fontweight="bold", color="#E64B35")

    bottom = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1], width_ratios=[1.7, 0.9], wspace=0.35
    )
    axis_curve = figure.add_subplot(bottom[0])
    expert_colors = ["#E64B35", "#4DBBD5", "#00A087", "#F39B7F"]
    for expert, color in zip(experts, expert_colors):
        axis_curve.plot(
            routing["sequence_position_bp"],
            routing[f"{expert}_rolling80_mean_probability"],
            color=color,
            linewidth=2.0,
            label=expert,
            alpha=0.9,
        )
    axis_curve.axvline(config["f2c"]["utr_boundary"], color="#111111", linestyle="--", linewidth=2)
    axis_curve.set_xlabel("Sequence Position (bp)", fontsize=12)
    axis_curve.set_ylabel("Mean Routing Probability", fontsize=12)
    axis_curve.set_title("Region-aligned expert routing along transcript", fontsize=13, fontweight="bold")
    axis_curve.set_xlim(0, config["f2c"]["curve_end"])
    axis_curve.set_xticks([0, 1000, 2000, 4000, 6000, 8000])
    axis_curve.legend(frameon=False, fontsize=10, loc="upper right")
    axis_curve.spines[["top", "right"]].set_visible(False)
    axis_curve.text(-0.07, 1.06, "C", transform=axis_curve.transAxes, fontsize=18, fontweight="bold")
    ytop = axis_curve.get_ylim()[1]
    axis_curve.text(500, ytop * 0.97, "5'UTR", ha="center", va="top", fontsize=10, fontweight="bold")
    axis_curve.text(4500, ytop * 0.97, "CDS + 3'UTR", ha="center", va="top", fontsize=10, fontweight="bold")

    axis_bar = figure.add_subplot(bottom[1])
    pivot = regions.pivot(index="expert", columns="region", values="mean_expert_probability").loc[experts]
    positions = np.arange(len(experts))
    width = 0.35
    utr = pivot["5'UTR"].to_numpy()
    cds = pivot["CDS + 3'UTR"].to_numpy()
    axis_bar.bar(positions - width / 2, utr, width=width, label="5'UTR", color="#38BDF8", edgecolor="#333333")
    axis_bar.bar(positions + width / 2, cds, width=width, label="CDS + 3'UTR", color="#FB923C", edgecolor="#333333")
    axis_bar.set_title("Region-specific expert specialization", fontsize=13, fontweight="bold", pad=15)
    axis_bar.set_xticks(positions, experts, fontsize=10, fontweight="bold")
    axis_bar.set_ylabel("Mean Expert Probability", fontsize=11, fontweight="bold")
    axis_bar.set_ylim(0, 0.65)
    axis_bar.legend(frameon=False, loc="upper right", fontsize=10)
    axis_bar.spines[["top", "right"]].set_visible(False)
    for index in range(len(experts)):
        axis_bar.text(index - width / 2, utr[index] + 0.01, f"{utr[index]:.2f}", ha="center", fontsize=9, fontweight="bold")
        axis_bar.text(index + width / 2, cds[index] + 0.01, f"{cds[index]:.2f}", ha="center", fontsize=9, fontweight="bold")

    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "InvaRNA_Fig2_Combined.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "InvaRNA_Fig2_Combined.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/fig2")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-hash-check", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    source_dir = args.source_dir or ROOT / config["source_dir"]
    hash_count = 0 if args.skip_hash_check else verify_hashes(source_dir)
    tables = load_tables(source_dir)
    summary = validate(tables, config, source_dir)
    summary["checksummed_files"] = hash_count
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fig2_validation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    if not args.check_only:
        render(tables, args.output_dir, config)
    print(
        "PASS: Fig. 2 source tables validated; "
        f"F2A rho={summary['f2a_centroid_spearman_rho']:.6f}; "
        f"F2C transcripts={summary['f2c_transcripts']}"
    )
    if not args.check_only:
        print(f"saved {args.output_dir / 'InvaRNA_Fig2_Combined.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
