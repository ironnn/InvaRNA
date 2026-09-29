#!/usr/bin/env python
"""Score engineered features with the frozen privileged teacher."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from invarna.distillation.teacher_scoring import main


if __name__ == "__main__":
    main()
