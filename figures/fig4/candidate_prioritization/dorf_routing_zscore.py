#!/usr/bin/env python3
"""Genome-wide dORF prioritization by native-vs-shuffled 3'UTR routing Z-score."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from invarna.data.fixed_frame import FIXED_CDS_START, TOTAL_LENGTH, align_and_pad
from invarna.evaluation.inference_api import InvaRNAPredictor
from invarna.interpretability.dorf_scoring import EXPERT_NAMES, extract_utr3, rank_dorf_candidates
from invarna.interpretability.feature_shuffle import expert_region_zscore
from invarna.interpretability.routing import infer_aligned_routing


UTR5_BACKBONE = "GGGAGACCCAAGCTGGCTAGCGTTTAAACTTAAGCTTGGTACCGAGCTCGGATCC"
CDS_BACKBONE = (
    "ATGGGAGTCAAAGTTCTGTTTGCCCTGATCTGCATCGCTGTGGCCGAGGCCAAGCCCACCGAGAACAACGAAGACTTCAACATCGTGGCCGTGGCCAGCAACTTCGCGACCACGGATCTCGATGCTGACCGCGGGAAGTTGCCCGGCAAGAAGCTGCCGCTGGAGGTGCTCAAAGAGATGGAAGCCAATGCCCGGAAAGCTGGCTGCACCAGGGGCTGTCTGATCTGCCTGTCCCACATCAAGTGCACGCCCAAGATGAAGAAGTTCATCCCAGGACGCTGCCACACCTACGAAGGCGACAAAGAGTCCGCACAGGGCGGCATAGGCGAGGCGATCGTCGACATTCCTGAGATTCCTGGGTTCAAGGACTTGGAGCCCATGGAGCAGTTCATCGCACAGGTCGATCTGTGTGTGGACTGCACAACTGGCTGCCTCAAAGGGCTTGCCAACGTGCAGTGTTCTGACCTGCTCAAGAAGTGGCTGCCGCAACGCTGTGCGACCTTTGCCAGCAAGATCCAGGGCCAGGTGGACAAGATCAAGGGGGCCGGTGGTGACTAA"
)
UTR3_START = FIXED_CDS_START + len(CDS_BACKBONE)


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)


def shuffled(sequence: str, rng: random.Random) -> str:
    chars = list(sequence)
    rng.shuffle(chars)
    return "".join(chars)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dorf-orfs", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank-output", type=Path)
    parser.add_argument("--model", default="final_tdc")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-permutations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    orfs = read_table(args.dorf_orfs).sort_values("orf_rpkm", ascending=False)
    unique = orfs.drop_duplicates("SYMBOL", keep="first")[["SYMBOL", "transcript_id", "gene_id"]]
    transcripts = read_table(args.transcripts)
    merged = unique.merge(
        transcripts[["transcript_id", "gene_id", "tx_sequence", "utr5_size", "cds_size"]],
        on=["transcript_id", "gene_id"], how="inner",
    )
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    quotient, remainder = divmod(len(merged), args.num_shards)
    start = args.shard_index * quotient + min(args.shard_index, remainder)
    end = start + quotient + int(args.shard_index < remainder)
    merged = merged.iloc[start:end]
    predictor = InvaRNAPredictor(args.model, device=args.device)
    rng = random.Random(args.seed + args.shard_index)
    rows = []
    for row in merged.itertuples(index=False):
        utr3 = extract_utr3(row.tx_sequence, row.utr5_size, row.cds_size).upper().replace("T", "U")
        if len(utr3) < 10:
            continue
        sequences = [utr3] + [shuffled(utr3, rng) for _ in range(args.n_permutations)]
        aligned = [align_and_pad(UTR5_BACKBONE + CDS_BACKBONE + seq, len(UTR5_BACKBONE)) for seq in sequences]
        routing = infer_aligned_routing(predictor, aligned, batch_size=args.batch_size)
        end = min(UTR3_START + len(utr3), TOTAL_LENGTH)
        result = expert_region_zscore(routing, UTR3_START, end)
        for index, expert in enumerate(EXPERT_NAMES):
            rows.append({
                "SYMBOL": row.SYMBOL, "transcript_id": row.transcript_id,
                "Expert": expert, "real_prob": result["native"][index],
                "bg_mean": result["background_mean"][index],
                "bg_std": result["background_std"][index],
                "Z_Score": result["z_score"][index], "utr3_len": len(utr3),
                "shard_index": args.shard_index,
            })
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    if args.rank_output:
        args.rank_output.parent.mkdir(parents=True, exist_ok=True)
        rank_dorf_candidates(output).to_csv(args.rank_output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
