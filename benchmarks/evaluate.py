#!/usr/bin/env python
"""Compute held-out or grouped external metrics from a prediction table."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score


def load(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix in {".parquet", ".pq"} else pd.read_csv(path)


def metrics(frame, truth, prediction):
    clean = frame[[truth, prediction]].dropna()
    return {"n": len(clean), "r2": r2_score(clean[truth], clean[prediction]),
            "spearman": spearmanr(clean[truth], clean[prediction]).statistic,
            "pearson": pearsonr(clean[truth], clean[prediction]).statistic}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--group", default=None, help="Species/library column")
    parser.add_argument("--truth-transform", choices=["none", "log10"], default="none")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frame = load(args.input)
    if args.truth_transform == "log10":
        if (frame[args.truth] <= 0).any():
            raise ValueError("log10 benchmark labels must be positive")
        frame = frame.copy()
        frame[args.truth] = np.log10(frame[args.truth])
    if args.group:
        result = pd.DataFrame([{"group": name, **metrics(group, args.truth, args.prediction)} for name, group in frame.groupby(args.group)])
    else:
        result = pd.DataFrame([metrics(frame, args.truth, args.prediction)])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
