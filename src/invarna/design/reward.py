"""Design objective: robust TE ensemble plus half-life.

This is not a complete PPO launcher. Predictor normalization, WT reference values,
and filter application must be supplied explicitly by the caller.
"""

import numpy as np


def robust_te_reward(
    z_scores, disagreement_weight: float = 0.15, disagreement_threshold: float = 0.5
) -> float:
    """Combine standardized TE-model improvements with a disagreement penalty."""
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
    """Complete public scoring objective; action constraints apply separately."""
    return te_weight * robust_te_reward(te_z_scores) + half_life_weight * float(half_life_z)
