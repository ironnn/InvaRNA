#!/usr/bin/env python3
"""Create an upload-ready hard-link copy of the frozen ``assets/`` tree.

The repository and large asset deposit use one shared layout: extracting the deposit
at the repository root populates ``assets/`` directly. This builder performs no
historical path mapping and never renames scientific artifacts.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets"
DEFAULT_TARGET = ROOT.parent / "InvaRNA_publication_assets_upload"

EXCLUDED_DIRS = {
    ".git",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
EXCLUDED_NAMES = {".DS_Store"}
EXCLUDED_PREFIXES = ("._",)
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".log")
EXCLUDED_RELATIVE_PREFIXES = ("checkpoints/external",)


def excluded(path: Path) -> bool:
    return (
        any(part in EXCLUDED_DIRS for part in path.parts)
        or any(path.as_posix().startswith(prefix) for prefix in EXCLUDED_RELATIVE_PREFIXES)
        or path.name in EXCLUDED_NAMES
        or path.name.startswith(EXCLUDED_PREFIXES)
        or path.name.endswith(EXCLUDED_SUFFIXES)
    )


def clone_assets(source: Path, destination: Path) -> tuple[int, int]:
    files = 0
    total_bytes = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if excluded(relative):
            continue
        target = destination / "assets" / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)
            files += 1
            total_bytes += path.stat().st_size
    return files, total_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()
    target = args.target.resolve()
    if not SOURCE.is_dir():
        parser.error(f"asset source is missing: {SOURCE}")
    if not (SOURCE / "MANIFEST.tsv").is_file() or not (SOURCE / "SHA256SUMS").is_file():
        parser.error("assets/MANIFEST.tsv and assets/SHA256SUMS must exist before bundling")
    if target.exists():
        parser.error(f"refusing to overwrite existing target: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        files, total_bytes = clone_assets(SOURCE, temporary)
        (temporary / "README.md").write_text(
            "# InvaRNA frozen assets\n\n"
            "Copy or extract the `assets/` directory into the code repository root, "
            "then run `python tests/verify_assets.py`.\n\n"
            "External baseline model weights are intentionally excluded. Their "
            "download sources, versions, licenses, and SHA-256 records are in "
            "`external_models/`.\n"
        )
        temporary.rename(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"created={target}")
    print(f"files={files}")
    print(f"bytes={total_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
