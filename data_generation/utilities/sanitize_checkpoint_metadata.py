#!/usr/bin/env python
"""Remove local absolute paths from Lightning checkpoint metadata only.

The utility refuses to replace a file unless a tensor-by-tensor state-dict
fingerprint is identical before and after serialization.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path

import torch


PRIVATE_PATH_PATTERNS = (
    re.compile(r"/home/[^/]+/"),
    re.compile(r"/Users/[^/]+/"),
    re.compile(r"/mnt/(?:data|files)\d*/[^/]+/"),
    re.compile(r"/workspace/volume/"),
)


def tensor_fingerprint(state_dict: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        digest.update(key.encode())
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


def scrub(value, replacement: str):
    if isinstance(value, str) and any(
        pattern.search(value) for pattern in PRIVATE_PATH_PATTERNS
    ):
        return replacement
    if isinstance(value, dict):
        return {key: scrub(item, replacement) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub(item, replacement) for item in value]
    if isinstance(value, tuple):
        return tuple(scrub(item, replacement) for item in value)
    return value


def sanitize(path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"{path}: missing Lightning state_dict")
    before = tensor_fingerprint(state_dict)
    checkpoint["hyper_parameters"] = scrub(
        checkpoint.get("hyper_parameters", {}), "outputs/external_te",
    )
    temporary = path.with_name(path.name + ".sanitized.tmp")
    torch.save(checkpoint, temporary)
    del checkpoint

    reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
    after = tensor_fingerprint(reloaded["state_dict"])
    del reloaded
    if before != after:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{path}: state_dict changed while sanitizing metadata")
    os.replace(temporary, path)
    print(f"PASS {path} state_dict_sha256={after}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()
    for checkpoint in args.checkpoints:
        sanitize(checkpoint)


if __name__ == "__main__":
    main()
