#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
python figures/fig5/plot_f5c.py
python figures/fig5/plot_f5d_checked.py
python figures/fig5/plot_f5e_checked.py
