#!/usr/bin/env python
"""Generate non-local UTR replacements from a frozen Flow-Matching pool."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from invarna.synthetic.flow_matching import main


if __name__ == "__main__":
    main()
