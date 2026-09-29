from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common import pending_source_data

if __name__ == "__main__": pending_source_data("Extended Figure 4")
