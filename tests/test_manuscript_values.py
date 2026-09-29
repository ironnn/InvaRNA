#!/usr/bin/env python
"""Check committed result summaries against manuscript-rounded values."""

from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "assets/manuscript_figures/results"


def compare(section, observed, expected, tolerance):
    failures = []
    for key, target in expected.items():
        if key not in observed:
            failures.append(f"{section}: missing {key}")
            continue
        delta = abs(float(observed[key]) - float(target))
        status = "PASS" if delta <= tolerance else "FAIL"
        print(f"{status} {section}/{key}: observed={observed[key]:.10g} manuscript={target:.10g} delta={delta:.3g}")
        if delta > tolerance:
            failures.append(f"{section}/{key}: delta={delta}")
    return failures


def main():
    expected = yaml.safe_load((ROOT / "benchmarks/configs/manuscript_results.yaml").read_text())
    tolerance = float(expected.pop("tolerance"))
    ablation = pd.read_csv(DATA / "fig3_ablation.csv").set_index("configuration")["test_r2"].to_dict()
    heldout = pd.read_csv(DATA / "fig3_heldout_human.csv").set_index("model")["heldout_human_te_r2"].to_dict()
    mpra = pd.read_csv(DATA / "fig3_mpra_summary.csv").set_index("model")["mean_spearman_rho"].to_dict()
    rpfd = pd.read_csv(DATA / "rpfd_final.csv").set_index("species")["spearman"].to_dict()
    checkpoint = pd.read_csv(DATA / "checkpoint_consistency.csv").set_index("comparison").loc[
        "pre_rename_original_vs_native"
    ]
    failures = []
    failures += compare("fig3_ablation", ablation, expected["fig3_ablation"], tolerance)
    failures += compare("fig3_heldout_human", heldout, expected["fig3_heldout_human"], tolerance)
    failures += compare("fig3_mpra_mean_spearman", mpra, expected["fig3_mpra_mean_spearman"], tolerance)
    failures += compare("rpfd_final", rpfd, expected["rpfd_final"], tolerance)
    failures += compare(
        "checkpoint_consistency",
        {
            "n": checkpoint["n"],
            "max_abs_prediction_difference": checkpoint["max_abs_prediction_difference"],
            "r2_difference": checkpoint["r2_difference"],
        },
        expected["checkpoint_consistency"],
        tolerance,
    )
    if failures:
        raise SystemExit("\n".join(failures))
    print("All committed manuscript-result checks passed.")


if __name__ == "__main__":
    main()
