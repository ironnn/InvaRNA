#!/usr/bin/env python
"""CLI wrapper for InvaRNA-native Stage1/Stage2 inference."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invarna.evaluation.human_te import main


if __name__ == "__main__":
    main()
