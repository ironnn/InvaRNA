#!/usr/bin/env python
"""Fail-fast leakage checks for human TE and synthetic anchor tables."""

import argparse
from pathlib import Path
import pandas as pd


def load(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix in {".parquet", ".pq"} else pd.read_csv(path)


def ids(frame, column):
    if column not in frame:
        raise ValueError(f"missing split column {column!r}")
    return set(frame[column].dropna().astype(str).str.replace(r"\.\d+$", "", regex=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--synthetic", action="append", default=[])
    parser.add_argument("--gene-col", default="gene_id")
    args = parser.parse_args()
    frames = {name: load(path) for name, path in (("train", args.train), ("validation", args.validation), ("test", args.test))}
    groups = {name: ids(frame, args.gene_col) for name, frame in frames.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = groups[left] & groups[right]
        if overlap:
            raise SystemExit(f"FAIL {left}/{right}: {len(overlap)} overlapping genes")
    for path in args.synthetic:
        synthetic = ids(load(path), args.gene_col)
        if not synthetic <= groups["train"]:
            raise SystemExit(f"FAIL {path}: {len(synthetic - groups['train'])} synthetic anchors are not in training genes")
        if synthetic & (groups["validation"] | groups["test"]):
            raise SystemExit(f"FAIL {path}: held-out genes occur in synthetic variants")
    counts = {name: len(group) for name, group in groups.items()}
    print(f"PASS gene-disjoint train/validation/test: {counts}")
    print(f"PASS {len(args.synthetic)} synthetic table(s): training anchors only")


if __name__ == "__main__":
    main()
