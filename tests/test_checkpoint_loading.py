"""Lightweight structural check for the canonical deployed TE checkpoint."""

from __future__ import annotations

import torch

from invarna.evaluation.human_te import CHECKPOINT_ROOT, _checkpoint_state


def test_canonical_te_checkpoint_deserializes() -> None:
    checkpoint = CHECKPOINT_ROOT / "te_student/final_tdc.ckpt"
    state = _checkpoint_state(checkpoint, torch.device("cpu"))
    assert isinstance(state, dict)
    assert len(state) == 357
