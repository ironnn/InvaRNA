#!/usr/bin/env python
"""Reproduce the visible Fig. 3g MPRA mean +/- SD bar from saved predictions."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "benchmarks/configs/fig3_panel_g.yaml"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def resolved(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def group_sample(frame: pd.DataFrame, group: str, n: int, seed: int) -> pd.DataFrame:
    """Match the historical groupby(...).apply(lambda x: x.sample(...)) call."""
    pieces = []
    for _, part in frame.groupby(group, sort=True):
        pieces.append(part.sample(n=min(n, len(part)), random_state=seed))
    return pd.concat(pieces, axis=0).reset_index(drop=True)


def verify_hashes(config: dict) -> None:
    sources = {
        "panel_3k": config["input"]["panel_3k"],
        "original_100k_per_library": config["input"]["original_100k_per_library"],
        "InvaRNA": config["models"]["InvaRNA"]["prediction_file"],
        "RiboNN": config["models"]["RiboNN"]["prediction_file"],
        "UTRLM": config["models"]["UTRLM"]["prediction_file"],
    }
    for name, value in sources.items():
        path = resolved(value)
        if not path.is_file():
            raise FileNotFoundError(f"Missing Fig. 3g source: {path}")
        observed = digest(path)
        expected = config["source_sha256"][name]
        if observed != expected:
            raise RuntimeError(f"SHA-256 mismatch for {name}: {observed} != {expected}")


def load_predictions(config: dict, overrides: dict[str, Path | None]) -> dict[str, pd.DataFrame]:
    frames = {}
    for model in ("InvaRNA", "RiboNN", "UTRLM"):
        path = overrides[model] or resolved(config["models"][model]["prediction_file"])
        frame = pd.read_csv(path)
        # The July 17 submission used the preserved 100k/library RiboNN and
        # UTR-LM files and sampled each library independently before dropping
        # missing truth/prediction pairs. A checkpoint rerun may already contain
        # only the 3k panel, in which case it must not be sampled again.
        if config["models"][model].get("apply_group_sampling", False):
            counts = frame.groupby(config["input"]["group_column"]).size()
            if counts.max() > int(config["sampling"]["per_library"]):
                frame = group_sample(
                    frame,
                    config["input"]["group_column"],
                    int(config["sampling"]["per_library"]),
                    int(config["sampling"]["random_state"]),
                )
        frames[model] = frame.reset_index(drop=True)
    return frames


def calculate(config: dict, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    group = config["input"]["group_column"]
    truth = config["input"]["truth_column"]
    libraries = sorted(frames["InvaRNA"][group].unique())
    rows = []
    for model in ("InvaRNA", "RiboNN", "UTRLM"):
        frame = frames[model]
        prediction = config["models"][model]["prediction_column"]
        for library in libraries:
            subset = frame[frame[group] == library].dropna(subset=[truth, prediction])
            if len(subset) <= 1:
                raise RuntimeError(f"Too few valid rows for {model}/{library}: {len(subset)}")
            rho = spearmanr(subset[truth], subset[prediction]).statistic
            rows.append({"model": model, "library": library, "n": len(subset), "spearman": rho})
    return pd.DataFrame(rows)


def check_expected(config: dict, summary: pd.DataFrame, tolerance: float = 1e-12) -> None:
    for row in summary.itertuples(index=False):
        expected = config["expected"][row.model]
        for field in ("mean", "sd"):
            observed = float(getattr(row, field))
            if not np.isclose(observed, float(expected[field]), rtol=0, atol=tolerance):
                raise RuntimeError(
                    f"Fig. 3g {row.model} {field} mismatch: {observed} != {expected[field]}"
                )


def plot(summary: pd.DataFrame, output: Path) -> None:
    order = ["UTRLM", "RiboNN", "InvaRNA"]
    palette = {"UTRLM": "#D1D5DB", "RiboNN": "#9CA3AF", "InvaRNA": "#DC2626"}
    values = summary.set_index("model").loc[order]
    x = np.arange(len(order))

    plt.rcdefaults()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axis = plt.subplots(figsize=(3.8, 4.2), dpi=300)
    axis.bar(
        x,
        values["mean"],
        yerr=values["sd"],
        capsize=6,
        width=0.55,
        color=[palette[name] for name in order],
        edgecolor="none",
        error_kw={"lw": 1.2},
        zorder=3,
    )
    for index, row in enumerate(values.itertuples()):
        axis.text(
            x[index], row.mean + row.sd + 0.015, f"{row.mean:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    axis.set_xticks(x, order, fontsize=10, fontweight="bold")
    axis.set_ylabel("Spearman ρ on external MPRA\n(mean ± SD, n=10)", fontsize=10)
    axis.set_ylim(0, 0.55)
    axis.axhline(0, color="black", linewidth=1.0)
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    axis.tick_params(axis="x", length=0)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/fig3_panel_g")
    parser.add_argument("--invarna-predictions", type=Path)
    parser.add_argument("--ribonn-predictions", type=Path)
    parser.add_argument("--utrlm-predictions", type=Path)
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument("--expected-tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    overrides = {
        "InvaRNA": args.invarna_predictions,
        "RiboNN": args.ribonn_predictions,
        "UTRLM": args.utrlm_predictions,
    }
    if not args.skip_hash_check and not any(overrides.values()):
        verify_hashes(config)
    per_library = calculate(config, load_predictions(config, overrides))
    summary = (
        per_library.groupby("model", sort=False)["spearman"]
        .agg(mean="mean", sd="std", libraries="count")
        .reset_index()
    )
    check_expected(config, summary, tolerance=args.expected_tolerance)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_library.to_csv(args.output_dir / "fig3_panel_g_per_library.csv", index=False)
    summary.to_csv(args.output_dir / "fig3_panel_g_summary.csv", index=False)
    plot(summary, args.output_dir / "fig3_panel_g.png")
    print(summary.to_string(index=False))
    print(f"PASS: reproduced Fig. 3g in {args.output_dir}")


if __name__ == "__main__":
    main()
