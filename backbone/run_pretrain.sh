#!/bin/bash
# =============================================================================
# InvaRNA backbone MLM pretraining — single-node multi-GPU launcher.
#
# Single-node launcher using the active Python interpreter.
#
# Full corpus (train0606.fasta = 106GB, val0606.fasta = 2.2GB) is referenced in
# the archival data deposit documented in assets/README.md.
# =============================================================================
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=${PYTHON:-python}
DATA="$ROOT/assets/manuscript_figures"

DEVICES=${DEVICES:-7}        # GPUs on this node
MAX_STEPS=${MAX_STEPS:-50000}

echo "[pretrain] $(date) start: devices=$DEVICES max_steps=$MAX_STEPS"
"$PY" "$ROOT/backbone/pretrain.py" \
    --train_file "$DATA/train0606.fasta" \
    --val_file   "$DATA/val0606.fasta" \
    --devices    "$DEVICES" \
    --num_nodes  1 \
    --max_steps  "$MAX_STEPS"
rc=$?
echo "[pretrain] $(date) exited rc=$rc"
