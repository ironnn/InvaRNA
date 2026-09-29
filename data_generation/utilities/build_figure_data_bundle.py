#!/usr/bin/env python
"""Assemble the frozen F2--F5 source tables in the canonical asset layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT.parent / "InvaRNA_publication_assets_upload" / "assets" / "manuscript_figures"
README_SOURCE = ROOT / "assets/manuscript_figures/full/README.md"

SOURCES = {
    "fig2_compact": ROOT / "assets/manuscript_figures/fig2_compact",
    "fig4_compact": ROOT / "assets/manuscript_figures/fig4_compact",
    "full/fig3_current_nm": ROOT / "assets/manuscript_figures/full/fig3_current_nm",
    "full/fig5": ROOT / "assets/manuscript_figures/full/fig5",
}


def link_tree(source: Path, destination: Path, manifest_root: Path) -> list[tuple[str, int]]:
    if not source.is_dir():
        raise FileNotFoundError(source)
    rows: list[tuple[str, int]] = []
    for src in sorted(path for path in source.rglob("*") if path.is_file()):
        rel = src.relative_to(source)
        dst = destination / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            same = src.stat().st_ino == dst.stat().st_ino and src.stat().st_dev == dst.stat().st_dev
            if src.stat().st_size != dst.stat().st_size:
                raise FileExistsError(f"refusing to replace size-mismatched file: {dst}")
            if not same and sha256(src) != sha256(dst):
                raise FileExistsError(f"refusing to replace content-mismatched file: {dst}")
        else:
            try:
                os.link(src, dst)
            except OSError:
                # Different filesystems: copy only when hard-linking is impossible.
                import shutil
                shutil.copy2(src, dst)
        rows.append((str(dst.relative_to(manifest_root)), src.stat().st_size))
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def register_with_parent_manifest(output: Path) -> None:
    bundle = output.parents[1]
    manifest = bundle / "MANIFEST.tsv"
    sums = bundle / "SHA256SUMS"
    if not manifest.is_file() or not sums.is_file():
        raise FileNotFoundError("parent bundle MANIFEST.tsv/SHA256SUMS not found")
    with manifest.open(newline="") as handle:
        existing = list(csv.DictReader(handle, delimiter="\t"))
    prefix = output.relative_to(bundle).as_posix() + "/"
    retained = {row["path"]: row for row in existing if not row["path"].startswith(prefix)}
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(bundle).as_posix()
        retained[relative] = {
            "path": relative,
            "bytes": str(path.stat().st_size),
            "sha256": sha256(path),
            "asset_group": "manuscript_figures",
        }
    ordered = [retained[key] for key in sorted(retained)]
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("path", "bytes", "sha256", "asset_group"), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(ordered)
    sums.write_text("".join(f"{row['sha256']}  {row['path']}\n" for row in ordered))
    print(f"registered {len(ordered)} total files in parent bundle manifest")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--register-parent-manifest", action="store_true",
        help="hash only the figure-data addendum and merge it into the existing asset manifest",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, int]] = []
    for relative, source in SOURCES.items():
        rows.extend(link_tree(source, output / relative, output.parent))

    manifest = output / "MANIFEST_PATH_SIZE.tsv"
    manifest.write_text(
        "path\tbytes\n" + "".join(f"{path}\t{size}\n" for path, size in sorted(rows))
    )
    readme = output / "README.md"
    if readme.exists():
        if readme.read_bytes() != README_SOURCE.read_bytes():
            raise FileExistsError(f"README differs from canonical source: {readme}")
    else:
        try:
            os.link(README_SOURCE, readme)
        except OSError:
            import shutil
            shutil.copy2(README_SOURCE, readme)
    print(f"assembled {len(rows)} files under {output}")
    print(f"manifest: {manifest}")
    if args.register_parent_manifest:
        register_with_parent_manifest(output)


if __name__ == "__main__":
    main()
