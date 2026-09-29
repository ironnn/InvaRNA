"""Auditable reward components recovered from the production design evaluator.

This is not a complete PPO launcher. Predictor normalization, WT reference values,
and filter application must be supplied explicitly by the caller.
"""

import numpy as np


def composition_penalty(sequence: str, gc_range=(0.45, 0.70), max_u=0.20) -> float:
    sequence = sequence.upper().replace("U", "T")
    length = max(len(sequence), 1)
    gc = (sequence.count("G") + sequence.count("C")) / length
    u = sequence.count("T") / length
    gc_distance = max(gc_range[0] - gc, 0.0, gc - gc_range[1])
    u_overflow = max(u - max_u, 0.0)
    return -100.0 * gc_distance**2 - 10.0 * u_overflow - 20.0 * u_overflow**2


def robust_te_reward(
    z_scores, disagreement_weight: float = 0.15, disagreement_threshold: float = 0.5
) -> float:
    """Combine standardized TE-model improvements using the recorded threshold rule."""
    scores = np.asarray(z_scores, dtype=float)
    disagreement = float(scores.std())
    penalty = disagreement_weight * disagreement if disagreement > disagreement_threshold else 0.0
    return float(scores.mean() - penalty)


def primary_reward(
    te_z_scores,
    half_life_z: float,
    te_weight: float = 0.70,
    half_life_weight: float = 0.30,
) -> float:
    """Recorded weighted TE/half-life objective before auxiliary terms and filters."""
    return te_weight * robust_te_reward(te_z_scores) + half_life_weight * float(half_life_z)
