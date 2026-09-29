#!/usr/bin/env python3
"""Numerically audit the source tables used in current manuscript Figure 5.

This validates archived values and construct identities; it does not regenerate the
experimental image panels or use experimental feedback in the RL workflow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import ttest_ind


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "assets/manuscript_figures/full/fig5"


def close(name: str, observed, expected, tolerance: float = 1e-9) -> float:
    error = float(np.nanmax(np.abs(np.asarray(observed, float) - np.asarray(expected, float))))
    if error > tolerance:
        raise RuntimeError(f"{name}: maximum absolute error {error} exceeds {tolerance}")
    return error


def audit_expression(data: Path, panel: str) -> dict[str, float]:
    raw = pd.read_csv(data / f"F5{panel}_expression_raw.csv")
    summary = pd.read_csv(data / f"F5{panel}_expression_summary.csv")
    value = "raw_signal" if panel == "A" else "protein_concentration_ng_ml"
    group_raw = "SeqID" if panel == "A" else "Variant"
    group_summary = "sample" if panel == "A" else "variant"
    grouped = raw.groupby(group_raw, sort=False)[value]
    means, sds, ns = grouped.mean(), grouped.std(), grouped.count()
    labels = summary[group_summary]
    if panel == "A":
        reference = {"blnk": "blnk_wt", "hbb": "hbb_wt"}
        expected_mean = []
        expected_sd = []
        for _, row in summary.iterrows():
            denominator = means[reference[row["gene"]]]
            expected_mean.append(means[row[group_summary]] / denominator)
            expected_sd.append(sds[row[group_summary]] / denominator)
        mean_error = close("F5A expression fold means", summary["mean_fold_change"], expected_mean)
        sd_error = close("F5A expression fold SD", summary["sd_fold_change"], expected_sd)
    else:
        mean_error = close("F5C expression means", summary["mean_ng_ml"], means[labels])
        sd_error = close("F5C expression SD", summary["sd_ng_ml"], sds[labels])
    if not np.array_equal(summary["n"].to_numpy(), ns[labels].to_numpy()):
        raise RuntimeError(f"F5{panel} expression replicate counts differ")
    return {"mean_max_abs_error": mean_error, "sd_max_abs_error": sd_error}


def audit_stability(data: Path, panel: str) -> dict[str, float]:
    raw = pd.read_csv(data / f"F5{panel}_stability_raw.csv")
    summary = pd.read_csv(data / f"F5{panel}_stability_summary.csv")
    fit = pd.read_csv(data / f"F5{panel}_stability_fit.csv")
    grouped = raw.groupby(["sample", "time_hours"], sort=False)["normalized_percent"]
    calculated = grouped.agg(["mean", "std", "count"]).reset_index()
    merged = summary.merge(calculated, on=["sample", "time_hours"], validate="one_to_one")
    nonzero = merged["time_hours"].ne(0)
    mean_error = close(
        f"F5{panel} stability means", merged.loc[nonzero, "mean_percent"], merged.loc[nonzero, "mean"]
    )
    sd_error = close(
        f"F5{panel} stability SD", merged.loc[nonzero, "sd_percent"], merged.loc[nonzero, "std"]
    )
    if not ((merged.loc[~nonzero, "mean_percent"] == 100).all()
            and (merged.loc[~nonzero, "sd_percent"] == 0).all()):
        raise RuntimeError(f"F5{panel} time-zero normalization differs from 100 +/- 0")
    if not np.array_equal(merged["n"].to_numpy(), merged["count"].to_numpy()):
        raise RuntimeError(f"F5{panel} stability replicate counts differ")

    fitted_k = []
    for sample in fit["sample"]:
        curve = summary.loc[summary["sample"].eq(sample)]
        parameter, _ = curve_fit(
            lambda time, k: 100.0 * np.exp(-k * time),
            curve["time_hours"].to_numpy(float),
            curve["mean_percent"].to_numpy(float),
            p0=[0.1],
        )
        fitted_k.append(parameter[0])
    k_error = close(f"F5{panel} fitted decay constants", fit["k_per_hour"], fitted_k, 1e-7)
    half_life_error = close(
        f"F5{panel} fitted half-lives", fit["half_life_hours"], np.log(2) / np.asarray(fitted_k), 2e-4
    )
    return {
        "mean_max_abs_error": mean_error,
        "sd_max_abs_error": sd_error,
        "k_max_abs_error": k_error,
        "half_life_max_abs_error": half_life_error,
    }


def audit_panel_d(data: Path) -> dict[str, float]:
    frame = pd.read_csv(data / "F5D_plot_data.csv")
    normalized = frame["flag"] / frame["tublin"]
    normalized_error = close("F5D normalized band intensity", frame["Normalized"], normalized)
    relative = []
    for _, row in frame.iterrows():
        reference = frame.loc[
            frame["batch"].eq(row["batch"]) & frame["display_label"].eq("Reference"), "Normalized"
        ].iloc[0]
        relative.append(row["Normalized"] / reference)
    relative_error = close("F5D relative band intensity", frame["Relative"], relative)
    return {"normalized_max_abs_error": normalized_error, "relative_max_abs_error": relative_error}


def audit_panel_e(data: Path) -> dict[str, float]:
    raw = pd.read_csv(data / "F5E_raw.csv")
    summary = pd.read_csv(data / "F5E_summary.csv")
    stats = pd.read_csv(data / "F5E_statistics.csv")
    grouped = raw.groupby(["magnification", "group"], sort=False)["rgc_count"]
    calculated = grouped.agg(["mean", "std", "count"])
    keys = list(zip(summary["magnification"], summary["group"]))
    mean_error = close("F5E means", summary["mean_rgc_count"], [calculated.loc[k, "mean"] for k in keys])
    sd_error = close("F5E SD", summary["sd_rgc_count"], [calculated.loc[k, "std"] for k in keys])
    if summary["n"].tolist() != [int(calculated.loc[k, "count"]) for k in keys]:
        raise RuntimeError("F5E replicate counts differ")
    pvalues = []
    for _, row in stats.iterrows():
        subset = raw.loc[raw["magnification"].eq(row["magnification"])]
        first = subset.loc[subset["group"].eq(row["group_1"]), "rgc_count"]
        second = subset.loc[subset["group"].eq(row["group_2"]), "rgc_count"]
        pvalues.append(ttest_ind(first, second, equal_var=True).pvalue)
    p_error = close("F5E t-test p-values", stats["p_value"], pvalues)
    return {"mean_max_abs_error": mean_error, "sd_max_abs_error": sd_error, "p_max_abs_error": p_error}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results/fig5_numeric_audit.json")
    args = parser.parse_args()
    data = args.data_dir or DEFAULT_DATA
    if not data.is_dir():
        raise SystemExit("Fig. 5 external source-data bundle is not installed")
    report = {
        "status": "PASS",
        "scope": "current manuscript Figure 5 numeric tables",
        "panel_A_expression": audit_expression(data, "A"),
        "panel_A_stability": audit_stability(data, "A"),
        "panel_B": {
            "status": "AUTHOR_CURATED_MANUAL_SELECTION",
            "reported_te_improved": 18,
            "reported_te_and_stability_improved": 4,
            "algorithmic_threshold": None,
        },
        "panel_C_expression": audit_expression(data, "C"),
        "panel_C_stability": audit_stability(data, "C"),
        "panel_D": audit_panel_d(data),
        "panel_E": audit_panel_e(data),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
