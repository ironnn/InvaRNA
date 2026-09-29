"""Validation utilities for translation-efficiency tables."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("mrna", "utr5_size")


def load_te_table(path, label_col: str | None = None) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_parquet(path) if path.suffix in {".parquet", ".pq"} else pd.read_csv(path)
    required = set(REQUIRED_COLUMNS)
    if label_col:
        required.add(label_col)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return frame
