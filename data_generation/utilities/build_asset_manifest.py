#!/usr/bin/env python3
"""Refresh the frozen-asset manifest without rehashing unchanged large files.

Files at or below 5 MiB are always hashed. For larger files, the previous manifest
hash is reused only when both path and byte size are unchanged, except for the
explicitly listed sub-GiB column-pruning rewrites. Files at or above 1 GiB are never
hashed by this utility. Any other new/changed large file stops the build, so an
expensive hash is never triggered accidentally.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
MANIFEST = ASSETS / "MANIFEST.tsv"
CHECKSUMS = ASSETS / "SHA256SUMS"
HASH_LIMIT = 5 * 1024 * 1024
EXCLUDED = {"MANIFEST.tsv", "SHA256SUMS"}
# External baseline weights are never part of the frozen publication asset
# deposit.  Their source URLs, artifact versions, licenses, and checksums live
# under ``external_models/`` instead.
EXCLUDED_PREFIXES = ("checkpoints/external/",)
EXCLUDED_NAMES = {".DS_Store"}
EXCLUDED_NAME_PREFIXES = ("._",)
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".log")
RENAMES = {
}
# Deliberate post-audit column-pruning rewrites. These files are below the user's
# 1-GiB no-rescan threshold, so their new digests are recomputed once here.
FORCE_HASH_PATHS = {
    "manuscript_figures/full/fig3_current_nm/F3A_TE_distribution_plot_values.csv.gz",
    "manuscript_figures/full/fig3_current_nm/F3B_three_model_predictions_merged.csv",
    "manuscript_figures/full/fig3_current_nm/F3B_species_spearman_no_fly.csv",
    "manuscript_figures/full/fig3_current_nm/F3C_UMAP_plot_points.csv",
    "manuscript_figures/full/fig3_current_nm/F3C_conservation_track_and_sampled_mutations.csv",
    "manuscript_figures/full/fig3_current_nm/F3F_heldout_human_TE_prediction_R2.csv",
    "manuscript_figures/full/fig3_current_nm/F3G_external_MPRA_spearman_by_sample.csv",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    previous: dict[str, tuple[int, str]] = {}
    if MANIFEST.is_file():
        with MANIFEST.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                previous[row["path"]] = (int(row["bytes"]), row["sha256"])

    rows: list[dict[str, str | int]] = []
    reused = hashed = 0
    for path in sorted(ASSETS.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ASSETS).as_posix()
        if (
            relative in EXCLUDED
            or any(relative.startswith(prefix) for prefix in EXCLUDED_PREFIXES)
            or "__pycache__" in path.parts
            or path.name in EXCLUDED_NAMES
            or path.name.startswith(EXCLUDED_NAME_PREFIXES)
            or path.name.endswith(EXCLUDED_SUFFIXES)
        ):
            continue
        size = path.stat().st_size
        old_path = RENAMES.get(relative, relative)
        old = previous.get(relative) or previous.get(old_path)
        if relative in FORCE_HASH_PATHS or size <= HASH_LIMIT:
            checksum = digest(path)
            hashed += 1
        elif old is not None and old[0] == size:
            checksum = old[1]
            reused += 1
        else:
            raise RuntimeError(
                f"new or resized large asset requires deliberate hashing: {relative} ({size} bytes)"
            )
        first = relative.split("/", 1)[0]
        group = first if "/" in relative else "metadata"
        rows.append({"path": relative, "bytes": size, "sha256": checksum, "asset_group": group})

    temporary_manifest = MANIFEST.with_suffix(".tsv.tmp")
    temporary_checksums = CHECKSUMS.with_suffix(".tmp")
    with temporary_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "bytes", "sha256", "asset_group"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    with temporary_checksums.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row['sha256']}  {row['path']}\n")
    temporary_manifest.replace(MANIFEST)
    temporary_checksums.replace(CHECKSUMS)
    print(f"assets={len(rows)} hashed_small={hashed} reused_large={reused}")
    print(f"bytes={sum(int(row['bytes']) for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
