"""Helpers for normalizing Ensembl transcript identifiers."""

import re

_VERSION_SUFFIX = re.compile(r"\.\d+$")
_MUTATION_SUFFIX = re.compile(r"_mut\d+$")


def canonical_transcript_id(value: str, strip_mutation: bool = False) -> str:
    """Strip an Ensembl version suffix and, optionally, an InvaRNA mutation suffix."""
    value = _VERSION_SUFFIX.sub("", str(value).strip())
    return _MUTATION_SUFFIX.sub("", value) if strip_mutation else value
