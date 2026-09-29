#!/usr/bin/env python3
"""Score F4A regulatory elements by local shuffle and MoE routing Z-score.

Input columns: ``Condition``, ``tx_sequence``, ``utr5_size``, ``element_start``,
``element_end``. Coordinates are zero-based, half-open transcript coordinates.
The data-specific uORF/RBP/miRNA coordinate rules are documented in
``docs/INTERPRETABILITY.md``.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from invarna.data.fixed_frame import TOTAL_LENGTH, align_and_pad
from invarna.evaluation.inference_api import InvaRNAPredictor
from invarna.interpretability.feature_shuffle import (
    EXPERT_NAMES,
    aligned_interval,
    expert_region_zscore,
    make_local_permutations,
)
from invarna.interpretability.routing import infer_aligned_routing


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="final_tdc")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-permutations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    frame = read_table(args.input)
    required = {"Condition", "tx_sequence", "utr5_size", "element_start", "element_end"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    bounds = [round(len(frame) * i / args.num_shards) for i in range(args.num_shards + 1)]
    frame = frame.iloc[bounds[args.shard_index]:bounds[args.shard_index + 1]]

    predictor = InvaRNAPredictor(args.model, device=args.device)
    rng = random.Random(args.seed)
    rows = []
    for index, row in frame.iterrows():
        aligned = align_and_pad(row.tx_sequence, row.utr5_size)
        start, end = aligned_interval(row.element_start, row.element_end, row.utr5_size)
        if not 0 <= start < end <= TOTAL_LENGTH:
            continue
        permutations = make_local_permutations(
            aligned, start, end, n_permutations=args.n_permutations, rng=rng
        )
        routing = infer_aligned_routing(predictor, permutations, batch_size=args.batch_size)
        result = expert_region_zscore(routing, start, end)
        for expert_index, expert in enumerate(EXPERT_NAMES):
            rows.append({
                "source_index": index,
                "Condition": row.Condition,
                "Expert": expert,
                "native_prob": result["native"][expert_index],
                "background_mean": result["background_mean"][expert_index],
                "background_std": result["background_std"][expert_index],
                "Z_Score": result["z_score"][expert_index],
                "n_permutations": args.n_permutations,
                "shard_index": args.shard_index,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
