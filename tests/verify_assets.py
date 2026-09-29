#!/usr/bin/env python
"""Verify the installed frozen-asset manifest without regenerating any data."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = ROOT / "assets"
QUICK_HASH_LIMIT = 5 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Hash every file, including the large publication assets.",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=DEFAULT_ASSET_ROOT,
        help="Asset directory to verify (default: repository assets/).",
    )
    args = parser.parse_args()
    asset_root = args.asset_root.resolve()
    manifest = asset_root / "MANIFEST.tsv"

    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    failures: list[str] = []
    hashed = 0
    for row in rows:
        path = asset_root / row["path"]
        if not path.is_file():
            failures.append(f"missing: {row['path']}")
            continue
        if path.stat().st_size != int(row["bytes"]):
            failures.append(f"size mismatch: {row['path']}")
            continue
        # The two backbone pretraining FASTA files are deliberately recorded with
        # size plus first/last-1-MiB fingerprints.  They are >1 GiB and are not
        # rehashed during publication verification.
        if str(row["sha256"]).startswith("UNHASHED_GT1G:"):
            continue
        if args.full or path.stat().st_size <= QUICK_HASH_LIMIT:
            hashed += 1
            if sha256(path) != row["sha256"]:
                failures.append(f"SHA-256 mismatch: {row['path']}")

    if failures:
        print("FAIL: asset verification")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    mode = "full" if args.full else f"quick (hash <= {QUICK_HASH_LIMIT} bytes)"
    print(f"PASS: {len(rows)} assets exist with expected sizes; hashed={hashed}; mode={mode}")


if __name__ == "__main__":
    main()
