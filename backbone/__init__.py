"""Reviewer-facing access to the InvaRNA self-supervised backbone.

The implementation is kept under ``src/invarna``.  Add that source tree when
the lightweight ``backbone.*`` compatibility package is imported directly
from a repository checkout (without requiring an editable install first).
"""

from pathlib import Path
import sys

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

__all__ = []
