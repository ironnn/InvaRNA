#!/usr/bin/env python
"""Run the frozen InvaRNA-derived half-life predictor."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invarna.evaluation.human_te import main


if __name__ == "__main__":
    if "--model" not in sys.argv:
        sys.argv.extend(["--model", "hl_new"])
    main()
