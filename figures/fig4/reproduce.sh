#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
python figures/fig4/audit.py --strict
python figures/fig4/reproduce.py
