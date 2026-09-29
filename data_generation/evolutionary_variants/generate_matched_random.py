#!/usr/bin/env python
"""Generate region-wise matched-random controls from a K80 table."""

import argparse
import re
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from invarna.synthetic.matched_random import mutate_matched

SUFFIX = re.compile(r"_mut\d+$")


def read(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix in {".parquet", ".pq"} else pd.read_pickle(path)


def region_diffs(wt, mutant):
    u5, cds = int(wt.utr5_size), int(wt.cds_size)
    return (
        sum(a != b for a, b in zip(wt.mrna[:u5], mutant.mrna[:u5])),
        sum(a != b for a, b in zip(wt.mrna[u5:u5 + cds], mutant.mrna[u5:u5 + cds])),
        sum(a != b for a, b in zip(wt.mrna[u5 + cds:], mutant.mrna[u5 + cds:])),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k80", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cds-mode", choices=["free", "strict"], default="free")
    args = parser.parse_args()
    frame = read(args.k80).copy()
    frame["anchor_id"] = frame["transcript_id"].astype(str).str.replace(SUFFIX, "", regex=True)
    source = frame.get("variant_source", pd.Series(index=frame.index, dtype=object))
    wild = frame[source.eq("wt")]
    if wild.empty and "mut_step" in frame:
        wild = frame[frame["mut_step"] == 0]
    if wild.empty or wild["anchor_id"].duplicated().any():
        raise ValueError("input must contain exactly one WT row per anchor")
    wt_map = wild.set_index("anchor_id")
    variants = frame[~frame.index.isin(wild.index)].copy()
    rows = []
    for offset, row in enumerate(variants.itertuples(index=False)):
        wt = wt_map.loc[row.anchor_id]
        u5d, cdsd, u3d = region_diffs(wt, row)
        record = row._asdict()
        record["mrna"] = mutate_matched(
            wt.mrna, int(wt.utr5_size), int(wt.cds_size), int(wt.utr3_size),
            u5d, cdsd, u3d, cds_mode=args.cds_mode, seed=args.seed + offset,
        )
        record["variant_source"] = "matched_random"
        rows.append(record)
    result = pd.concat([wild.assign(variant_source="wt"), pd.DataFrame(rows)], ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    print(f"saved {len(result)} rows to {output}")


if __name__ == "__main__":
    main()
