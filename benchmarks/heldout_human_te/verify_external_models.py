#!/usr/bin/env python
"""Recompute held-out test R2 from the prediction files of selected LLM runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "assets/manuscript_figures/results/external_llm_predictions"


def read_predictions(path: Path) -> tuple[list[float], list[float]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"pred", "target"}.issubset(rows[0]):
        raise ValueError(f"{path} must contain pred,target columns")
    pred = [float(row["pred"]) for row in rows]
    target = [float(row["target"]) for row in rows]
    return pred, target


def r2_score(target: list[float], pred: list[float]) -> float:
    """Equivalent to sklearn.metrics.r2_score for one finite-output vector."""
    mean_target = sum(target) / len(target)
    residual = sum((truth - estimate) ** 2 for truth, estimate in zip(target, pred))
    total = sum((truth - mean_target) ** 2 for truth in target)
    if total == 0:
        raise ValueError("R2 is undefined for a constant target")
    return 1.0 - residual / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=5e-4)
    args = parser.parse_args()

    expected = {
        "codonbert": 0.5171,
        "dnabert2": 0.5371,
        "evo2_8k": 0.7303000306442183,
        "lucaone": 0.4552,
        "mrnabert": 0.3113,
        "orthrus": 0.5635,
        "ribonn": 0.6164582193177239,
        "rnafm": 0.3885,
        "utrlm": 0.1674,
        "invarna_human_mouse": 0.7061,
    }
    results = []
    failed = False
    for model, target_r2 in expected.items():
        pred, target = read_predictions(args.input_dir / f"{model}.csv")
        observed = float(r2_score(target, pred))
        passed = abs(observed - target_r2) <= args.tolerance
        failed |= not passed
        results.append((model, len(pred), observed, target_r2, passed))
        print(f"{model:22s} n={len(pred):4d} R2={observed:.6f} expected={target_r2:.6f} {'PASS' if passed else 'FAIL'}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model", "n", "observed_r2", "expected_r2", "passed"])
            writer.writerows(results)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
