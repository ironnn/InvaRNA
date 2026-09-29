"""Unit checks for the manuscript label-construction rules."""

from __future__ import annotations

import pandas as pd

from invarna.distillation.tdc import construct_labels


def test_k80_is_anchored_but_flow_matching_and_matched_random_are_absolute() -> None:
    frame = pd.DataFrame(
        [
            ("g_mut0", "AAAA", 1.0, 1.5, "wt"),
            ("g_mut1", "AAAT", 1.2, 1.5, "k80"),
            ("g_mut201", "TTTT", 2.0, 1.5, "flow_matching"),
            ("g_mut3", "AATA", 0.8, 1.5, "matched_random"),
        ],
        columns=["transcript_id", "mrna", "pred_score", "mean_te", "variant_source"],
    )
    labels = construct_labels(frame, alpha=1.0).set_index("transcript_id")
    assert labels.loc["g_mut0", "training_label"] == 1.5
    assert labels.loc["g_mut1", "training_label"] == 1.7
    assert labels.loc["g_mut201", "training_label"] == 2.0
    assert labels.loc["g_mut3", "training_label"] == 0.8
