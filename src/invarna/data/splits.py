"""Leakage-safe transcript split helpers."""

from collections.abc import Iterable

import pandas as pd

from .ensembl import canonical_transcript_id


def exclude_transcript_families(frame: pd.DataFrame, held_out_ids: Iterable[str]) -> pd.DataFrame:
    """Remove WT and mutant rows belonging to any held-out transcript family."""
    held_out = {canonical_transcript_id(value, strip_mutation=True) for value in held_out_ids}
    families = frame["transcript_id"].map(lambda x: canonical_transcript_id(x, True))
    return frame.loc[~families.isin(held_out)].copy()
