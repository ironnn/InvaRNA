#!/usr/bin/env python
"""Generate the manuscript K80-derived local variants."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from invarna.synthetic.k80 import main


if __name__ == "__main__":
    main()
