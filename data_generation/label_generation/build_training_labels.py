#!/usr/bin/env python3
"""Build final labels using the exact regime-aware TDC implementation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from invarna.distillation.tdc import main


if __name__ == "__main__":
    main()
