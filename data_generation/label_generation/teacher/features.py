"""Public import surface for the copied privileged-teacher feature extractor.

The underlying source is intentionally kept byte-for-byte in ``source/utils``.
"""

from __future__ import annotations

from pathlib import Path
import sys


SOURCE_UTILS = Path(__file__).resolve().parent / "source" / "utils"
sys.path.insert(0, str(SOURCE_UTILS))

from lgbm_feature_extract_from_str import *  # noqa: E402,F401,F403
