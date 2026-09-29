#!/usr/bin/env python
"""Fail if runnable repository files contain machine-specific absolute paths."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (
    "backbone",
    "benchmarks",
    "data_generation",
    "external_models",
    "figures",
    "inference",
    "sft",
    "src",
    "tests",
)
TEXT_SUFFIXES = {
    ".json",
    ".ipynb",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
PRIVATE_ROOT = re.compile(
    r"/(?:home|Users|mnt|scratch|fs\d+|ssd\d+|workspace|lustre|gpfs|data\d+)/"
)
PATH_KEY = re.compile(
    r"(?:path|file|dir|root|checkpoint|ckpt|repository|run_path)$", re.IGNORECASE
)

# This utility must recognize private path forms in order to remove them from
# checkpoint metadata. The strings are regular expressions, not runtime paths.
TEXT_EXCEPTIONS = {
    Path("data_generation/utilities/sanitize_checkpoint_metadata.py"),
    Path("tests/test_relative_paths.py"),
}

# Docker Compose requires an absolute working directory inside the container;
# this is not a host/repository path.
CONFIG_EXCEPTIONS = {
    Path("external_models/RiboNN/docker-compose.yml"),
}


def iter_files():
    for root_name in SCAN_ROOTS:
        root = ROOT / root_name
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                yield path


def walk_config(value, location=""):
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{location}.{key}" if location else str(key)
            if isinstance(item, str) and PATH_KEY.search(str(key)):
                if Path(item).is_absolute():
                    yield child, item
            yield from walk_config(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk_config(item, f"{location}[{index}]")


def main():
    failures = []
    checked = 0
    for path in iter_files():
        relative = path.relative_to(ROOT)
        checked += 1
        text = path.read_text(errors="replace")
        if relative not in TEXT_EXCEPTIONS:
            for match in PRIVATE_ROOT.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                failures.append(f"{relative}:{line}: {match.group(0)}")

        if path.suffix in {".yaml", ".yml"} and relative not in CONFIG_EXCEPTIONS:
            try:
                config = yaml.safe_load(text)
            except yaml.YAMLError:
                # Some copied upstream YAML fragments require their own loader.
                continue
            for key, value in walk_config(config):
                failures.append(f"{relative}:{key}: absolute value {value!r}")

    if failures:
        raise SystemExit("Machine-specific absolute paths found:\n" + "\n".join(failures))
    print(f"PASS relative-path audit: {checked} runnable text/config files")


if __name__ == "__main__":
    main()
