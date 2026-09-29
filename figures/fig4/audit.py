#!/usr/bin/env python3
"""Verify copied Fig. 4 artifacts and recompute compact consistency checks."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from invarna.interpretability.feature_shuffle import iqr_filter_element_scores, summarize_element_scores


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "assets" / "manuscript_figures" / "fig4_compact"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    failures = 0
    for line in (DATA / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        candidate = DATA / Path(relative).name
        if digest(candidate) != expected:
            print(f"FAIL checksum: {relative}")
            failures += 1

    local = pd.concat([
        pd.read_csv(DATA / "_f4a_local_s0.csv"),
        pd.read_csv(DATA / "_f4a_local_s1.csv"),
    ], ignore_index=True)
    local["Expert"] = local["Expert"].replace("Single Base", "Single-base")
    conditions = ["uORF", "RBP (5′UTR)", "RBP (CDS)", "RBP (3′UTR)", "miRNA"]
    experts = ["Single-base", "Codon", "9-mer", "Ribosome"]
    observed = summarize_element_scores(iqr_filter_element_scores(local), conditions, experts)
    expected = pd.read_csv(DATA / "fig4a_statistics_iqr.csv")
    merged = expected.merge(observed, on=["Condition", "Expert"], suffixes=("_saved", "_recomputed"))
    mean_delta = float(np.max(np.abs(merged["Mean_Z_saved"] - merged["Mean_Z_recomputed"])))
    n_equal = bool((merged["N_saved"] == merged["N_recomputed"]).all())

    candidate = pd.read_csv(DATA / "dorf_moe_summary_InvaRNA0412.csv")
    z_recomputed = (candidate["real_prob"] - candidate["bg_mean"]) / candidate["bg_std"]
    candidate_z_delta = float(np.max(np.abs(z_recomputed - candidate["z_score"])))
    genome = pd.read_csv(DATA / "dorf_zscore_genome_wide_InvaRNA0412.csv")
    orfs = pd.read_csv(DATA / "dorf_orfs.csv")

    checks = {
        "checksum_failures": failures,
        "f4a_element_rows": len(local),
        "f4a_condition_count": local["Condition"].nunique(),
        "f4a_summary_max_abs_delta": mean_delta,
        "f4a_summary_n_equal": n_equal,
        "dorf_candidate_summary_rows": len(candidate),
        "dorf_candidate_z_rounding_max_abs_delta": candidate_z_delta,
        "dorf_genomewide_genes": genome["SYMBOL"].nunique(),
        "dorf_orf_rows": len(orfs),
        "dorf_orf_genes": orfs["SYMBOL"].nunique(),
    }
    for key, value in checks.items():
        print(f"{key}={value}")
    failures += int(len(local) != 20000 or local["Condition"].nunique() != 5)
    failures += int(mean_delta > 1e-12 or not n_equal)
    failures += int(candidate_z_delta > 2e-4)
    failures += int(genome["SYMBOL"].nunique() != 8241 or orfs["SYMBOL"].nunique() != 8241)
    print(f"invariant_failures={failures}")
    return int(bool(args.strict and failures))


if __name__ == "__main__":
    raise SystemExit(main())
