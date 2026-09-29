"""Lightweight structural check for the canonical deployed TE checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from invarna.evaluation.human_te import _checkpoint_state


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_te_checkpoint_deserializes() -> None:
    checkpoint = ROOT / "assets/checkpoints/te_student/final_tdc.ckpt"
    state = _checkpoint_state(checkpoint, torch.device("cpu"))
    assert isinstance(state, dict)
    assert len(state) == 357
