#!/usr/bin/env python3
"""WT-anchored Taylor difference correction used for K80 local variants."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from invarna.distillation.tdc import *  # noqa: F403
from invarna.distillation.tdc import main


if __name__ == "__main__":
    main()
