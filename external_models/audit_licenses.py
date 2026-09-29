#!/usr/bin/env python
"""Audit copied baseline license evidence and enforce the public-release gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "external_models/THIRD_PARTY_LICENSES.tsv"
DOWNLOADS = ROOT / "external_models/DOWNLOADS.tsv"
CHECKSUMS = ROOT / "external_models/CHECKSUMS.tsv"
NOTICE = ROOT / "docs/THIRD_PARTY_LICENSES.md"
RESTRICTED_CLASS = "RESTRICTED_NONCOMMERCIAL"
DEFAULT_ASSET_ROOT = ROOT / "assets"

ASSET_LICENSE_PATHS = (
    "provenance/THIRD_PARTY_LICENSES.md",
    "provenance/THIRD_PARTY_LICENSES.tsv",
)


def read_inventory() -> list[dict[str, str]]:
    with INVENTORY.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise SystemExit(f"empty inventory: {INVENTORY}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def audit_asset_bundle(asset_root: Path, failures: list[str]) -> None:
    manifest_path = asset_root / "MANIFEST.tsv"
    sums_path = asset_root / "SHA256SUMS"
    if not manifest_path.is_file() or not sums_path.is_file():
        failures.append(f"asset bundle lacks MANIFEST.tsv/SHA256SUMS: {asset_root}")
        return
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest = {
            row["path"]: row for row in csv.DictReader(handle, delimiter="\t")
        }
    sums = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        sums[relative] = digest

    for relative in ASSET_LICENSE_PATHS:
        path = asset_root / relative
        row = manifest.get(relative)
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"asset bundle: missing/empty {relative}")
            continue
        observed = sha256(path)
        if row is None:
            failures.append(f"asset bundle: {relative} missing from MANIFEST.tsv")
            continue
        if int(row["bytes"]) != path.stat().st_size or row["sha256"] != observed:
            failures.append(f"asset bundle: stale manifest entry for {relative}")
        if sums.get(relative) != observed:
            failures.append(f"asset bundle: stale SHA256SUMS entry for {relative}")


def audit_external_metadata(failures: list[str]) -> None:
    """Require download/version/license metadata without requiring local weights."""
    try:
        with DOWNLOADS.open(encoding="utf-8", newline="") as handle:
            downloads = list(csv.DictReader(handle, delimiter="\t"))
    except OSError as error:
        failures.append(f"missing external download metadata: {error}")
        downloads = []
    required = {
        "component",
        "artifact_or_run",
        "source_url",
        "version_or_revision",
        "license",
        "local_layout",
    }
    for row in downloads:
        if not required.issubset(row) or any(not row[name].strip() for name in required):
            failures.append("external download metadata has an incomplete row")
            break
    if not downloads:
        failures.append("external download metadata is empty")

    try:
        with CHECKSUMS.open(encoding="utf-8", newline="") as handle:
            checksums = list(csv.DictReader(handle, delimiter="\t"))
    except OSError as error:
        failures.append(f"missing external checksum metadata: {error}")
        checksums = []
    if not checksums:
        failures.append("external checksum metadata is empty")
    for row in checksums:
        digest = row.get("sha256", "")
        if (
            not row.get("path")
            or not row.get("bytes", "").isdigit()
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            failures.append("external checksum metadata has an invalid row")
            break


def audit(strict_public: bool, asset_root: Path | None) -> int:
    failures: list[str] = []
    restricted: list[str] = []
    rows = read_inventory()
    audit_external_metadata(failures)

    if not NOTICE.is_file():
        failures.append("missing docs/THIRD_PARTY_LICENSES.md")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    if "third-party code" not in license_text.casefold():
        failures.append("top-level LICENSE lacks the third-party scope notice")

    for row in rows:
        component = row["component"]
        local_path = ROOT / row["local_path"]
        if not local_path.exists():
            failures.append(f"{component}: missing local path {row['local_path']}")
        evidence = [item for item in row["local_license_evidence"].split(";") if item]
        if not evidence:
            failures.append(f"{component}: no local license evidence declared")
        for relative in evidence:
            path = ROOT / relative
            if not path.is_file() or path.stat().st_size == 0:
                failures.append(f"{component}: missing/empty evidence {relative}")
        if row["redistribution_class"] == RESTRICTED_CLASS:
            restricted.append(component)

    if asset_root is not None:
        if asset_root.exists():
            audit_asset_bundle(asset_root, failures)
        else:
            failures.append(f"requested asset bundle does not exist: {asset_root}")

    if failures:
        print("FAIL: third-party license inventory is incomplete")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(
        f"PASS: {len(rows)} license records have local evidence; "
        f"restricted records={len(restricted)}"
    )
    if restricted:
        print("Restricted/non-commercial components:")
        for component in restricted:
            print(f"- {component}")

    if strict_public and restricted:
        print(
            "FAIL --strict-public: exclude the restricted artifacts from the "
            "public distribution or obtain redistribution permission."
        )
        return 2
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-public",
        action="store_true",
        help="Fail if any non-commercial/restricted component remains in scope.",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        help=(
            "Validate license files and their checksums in an asset bundle. If omitted, "
            "the standard sibling bundle is checked when it exists."
        ),
    )
    args = parser.parse_args()
    asset_root = args.asset_root
    if asset_root is None and DEFAULT_ASSET_ROOT.exists():
        asset_root = DEFAULT_ASSET_ROOT
    raise SystemExit(audit(args.strict_public, asset_root))


if __name__ == "__main__":
    main()
