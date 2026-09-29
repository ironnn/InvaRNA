#!/usr/bin/env python3
"""Nominate 3'UTR dORFs with the recovered ORFfinder + RPFdb rule."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from invarna.interpretability.dorf_scoring import (
    compute_median_utr3_rpkm,
    expand_candidate_orfs,
    filter_riboseq_supported_dorfs,
    nominate_dorf_transcripts,
)


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-orf-length", type=int, default=75)
    parser.add_argument("--counts-matrix", type=Path,
                        help="CSV: first column gene_base, remaining columns raw counts per sample")
    parser.add_argument("--gene-lengths", type=Path,
                        help="CSV with gene_base and length columns")
    parser.add_argument("--candidates-output", type=Path)
    parser.add_argument("--orfs-output", type=Path)
    args = parser.parse_args()
    transcripts = read_table(args.transcripts)
    annotations, orfs_by_transcript = nominate_dorf_transcripts(transcripts, args.minimum_orf_length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    annotations.to_csv(args.output, index=False)
    rpkm_options = (args.counts_matrix, args.gene_lengths, args.candidates_output, args.orfs_output)
    if any(rpkm_options) and not all(rpkm_options):
        parser.error("RPFdb filtering requires --counts-matrix, --gene-lengths, --candidates-output and --orfs-output")
    if all(rpkm_options):
        counts = pd.read_csv(args.counts_matrix, index_col=0)
        lengths_frame = pd.read_csv(args.gene_lengths)
        lengths = lengths_frame.set_index("gene_base")["length"].reindex(counts.index)
        median, fraction = compute_median_utr3_rpkm(counts, lengths)
        candidates = filter_riboseq_supported_dorfs(annotations, median, fraction)
        args.candidates_output.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(args.candidates_output, index=False)
        expanded = expand_candidate_orfs(candidates, transcripts, orfs_by_transcript)
        args.orfs_output.parent.mkdir(parents=True, exist_ok=True)
        expanded.to_csv(args.orfs_output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
