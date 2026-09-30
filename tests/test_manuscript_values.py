#!/usr/bin/env python
"""Check current Fig. 3 source tables against manuscript-rounded values.

Older RPFdb and checkpoint-consistency summaries can be checked separately when
the non-release result bundle is available.
"""

import argparse
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "assets/manuscript_figures/full/fig3_current_nm"
ABLATION_NAMES = {
    "human": "human",
    "human_mouse": "human_mouse",
    "rand_mut": "matched_random_absolute_teacher",
    "evol_mut": "k80_absolute_teacher",
    "taylor": "k80_wt_anchor",
    "final": "final_tdc",
}


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-results-dir", type=Path,
        help="Optional directory containing rpfd_final.csv and checkpoint_consistency.csv",
    )
    args = parser.parse_args()
    expected = yaml.safe_load((ROOT / "benchmarks/configs/manuscript_results.yaml").read_text())
    tolerance = float(expected.pop("tolerance"))
    ablation_source = pd.read_csv(DATA / "F3E_ablation_plot_data.csv")
    ablation = {
        ABLATION_NAMES[row.configuration]: row.test_r2_plotted
        for row in ablation_source.itertuples(index=False)
    }
    heldout = pd.read_csv(DATA / "F3F_heldout_human_TE_prediction_R2.csv").set_index(
        "model"
    )["heldout_human_te_prediction_r2"].to_dict()
    mpra_source = pd.read_csv(DATA / "F3G_external_MPRA_spearman_by_sample.csv")
    mpra = mpra_source.groupby("model")["spearman_rho"].mean().rename(
        index={"UTRLM": "UTR-LM"}
    ).to_dict()
    failures = []
    failures += compare("fig3_ablation", ablation, expected["fig3_ablation"], tolerance)
    failures += compare("fig3_heldout_human", heldout, expected["fig3_heldout_human"], tolerance)
    failures += compare("fig3_mpra_mean_spearman", mpra, expected["fig3_mpra_mean_spearman"], tolerance)
    if args.legacy_results_dir:
        rpfd = pd.read_csv(args.legacy_results_dir / "rpfd_final.csv").set_index("species")["spearman"].to_dict()
        checkpoint = pd.read_csv(args.legacy_results_dir / "checkpoint_consistency.csv").set_index("comparison").loc[
            "pre_rename_original_vs_native"
        ]
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
    else:
        print("SKIP RPFdb and checkpoint-consistency summaries (supply --legacy-results-dir)")
    if failures:
        raise SystemExit("\n".join(failures))
    print("Current-manuscript Fig. 3 value checks passed.")


if __name__ == "__main__":
    main()
