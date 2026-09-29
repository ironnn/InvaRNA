"""Fig. 4 local-element permutation and expert-routing statistics.

The core operations are copied from the final F4A implementation. File paths and
GPU selection are intentionally left to command-line callers.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats


EXPERT_NAMES = ("Single Base", "Codon", "9-mer", "Ribosome")


def shuffle_feature(values, seed: int = 42):
    result = np.asarray(values).copy()
    np.random.default_rng(seed).shuffle(result)
    return result


def shuffle_region(sequence: str, start: int, end: int, rng: random.Random) -> str:
    """Shuffle ``sequence[start:end]`` while preserving its nucleotide content."""
    if not 0 <= start < end <= len(sequence):
        raise ValueError(f"invalid half-open interval [{start}, {end}) for length {len(sequence)}")
    chars = list(sequence)
    region = chars[start:end]
    rng.shuffle(region)
    chars[start:end] = region
    return "".join(chars)


def make_local_permutations(
    aligned_sequence: str,
    region_start: int,
    region_end: int,
    n_permutations: int = 20,
    seed: int = 42,
    rng: random.Random | None = None,
) -> list[str]:
    """Return native sequence followed by the local-shuffle null sequences."""
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    rng = rng or random.Random(seed)
    return [aligned_sequence] + [
        shuffle_region(aligned_sequence, region_start, region_end, rng)
        for _ in range(n_permutations)
    ]


def expert_region_zscore(
    routing_probabilities: np.ndarray,
    region_start: int,
    region_end: int,
    epsilon: float = 1e-9,
) -> dict[str, np.ndarray]:
    """Compute the exact F4A native-versus-shuffled expert Z-score.

    ``routing_probabilities`` has shape ``[1 + permutations, length, experts]``;
    row zero is native and remaining rows are the permutation background.
    """
    probs = np.asarray(routing_probabilities, dtype=np.float64)
    if probs.ndim != 3 or probs.shape[0] < 2:
        raise ValueError("routing probabilities must have shape [native+permutations, length, experts]")
    if not 0 <= region_start < region_end <= probs.shape[1]:
        raise ValueError("routing interval is outside the model input")
    region_means = probs[:, region_start:region_end, :].mean(axis=1)
    native = region_means[0]
    background = region_means[1:]
    background_mean = background.mean(axis=0)
    background_std = background.std(axis=0)
    return {
        "native": native,
        "background": background,
        "background_mean": background_mean,
        "background_std": background_std,
        "z_score": (native - background_mean) / (background_std + epsilon),
    }


def aligned_interval(tx_start: int, tx_end: int, utr5_size: int, cds_start: int = 1000) -> tuple[int, int]:
    """Map a zero-based, half-open transcript interval into the fixed model frame."""
    left_padding = max(cds_start - int(utr5_size), 0)
    return left_padding + int(tx_start), left_padding + int(tx_end)


def uorf_interval(utr5_size: int, cds_distance: int, uorf_length: int) -> tuple[int, int]:
    """Recover the uORF interval used by F4A from distance-to-CDS and length."""
    end = int(utr5_size) - int(cds_distance)
    return end - int(uorf_length), end


def iqr_filter_element_scores(frame: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    """Apply the final F4A condition-by-expert IQR filter."""
    required = {"Condition", "Expert", "Z_Score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    grouped = frame.groupby(["Condition", "Expert"])["Z_Score"]
    q1 = grouped.transform(lambda values: values.quantile(0.25))
    q3 = grouped.transform(lambda values: values.quantile(0.75))
    iqr = q3 - q1
    keep = frame["Z_Score"].between(q1 - multiplier * iqr, q3 + multiplier * iqr)
    return frame.loc[keep].copy()


def summarize_element_scores(
    frame: pd.DataFrame,
    condition_order: Iterable[str] | None = None,
    expert_order: Iterable[str] = EXPERT_NAMES,
) -> pd.DataFrame:
    """Mean Z-score and one-sample two-sided t-test against zero, as in final F4A."""
    conditions = list(condition_order) if condition_order is not None else list(frame["Condition"].unique())
    records = []
    for condition in conditions:
        for expert in expert_order:
            values = frame.loc[
                (frame["Condition"] == condition) & (frame["Expert"] == expert), "Z_Score"
            ].dropna()
            p_value = float(stats.ttest_1samp(values, popmean=0.0).pvalue) if len(values) >= 5 else np.nan
            significance = (
                "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else
                "ns" if np.isfinite(p_value) else ""
            )
            records.append({
                "Condition": condition, "Expert": expert,
                "Mean_Z": float(values.mean()) if len(values) else np.nan,
                "P_Value": p_value, "Significance": significance, "N": int(len(values)),
            })
    return pd.DataFrame(records)
