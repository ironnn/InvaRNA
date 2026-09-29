"""Construct manuscript label targets without crossing variant regimes."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd

_MUT_SUFFIX = re.compile(r"_mut\d+$")
VALID_SOURCES = {"wt", "k80", "flow_matching", "matched_random"}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix in {".parquet", ".pq"} else pd.read_csv(path)


def _base_id(value: object) -> str:
    return _MUT_SUFFIX.sub("", str(value))


def construct_labels(frame: pd.DataFrame, *, alpha: float = 1.0) -> pd.DataFrame:
    """Apply measured-WT, K80-TDC, FM-absolute, and matched-absolute rules.

    An explicit ``variant_source`` is required to prevent applying TDC to the
    non-local Flow-Matching or matched-random control rows.
    """
    required = {"variant_source", "transcript_id", "mrna", "pred_score", "mean_te"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    unknown = set(frame["variant_source"].dropna().unique()) - VALID_SOURCES
    if unknown or frame["variant_source"].isna().any():
        raise ValueError(f"invalid variant_source values: {sorted(unknown)}")

    out = frame.copy()
    out["anchor_id"] = out["transcript_id"].map(_base_id)
    wt = out[out["variant_source"] == "wt"]
    if wt["anchor_id"].duplicated().any():
        raise ValueError("each anchor_id must have exactly one WT row")
    anchors = wt.set_index("anchor_id")
    for source, target in (("mrna", "anchor_sequence"), ("pred_score", "anchor_teacher"), ("mean_te", "anchor_measured")):
        out[target] = out["anchor_id"].map(anchors[source])
    if out[["anchor_sequence", "anchor_teacher", "anchor_measured"]].isna().any().any():
        raise ValueError("one or more variants have no WT anchor")

    def distance(row) -> int:
        sequence, anchor = str(row["mrna"]), str(row["anchor_sequence"])
        if len(sequence) != len(anchor):
            raise ValueError(f"length mismatch for {row['transcript_id']}")
        return sum(a != b for a, b in zip(sequence, anchor))

    out["n_mut"] = 0
    local = out["variant_source"].isin({"k80", "matched_random"})
    out.loc[local, "n_mut"] = out.loc[local].apply(distance, axis=1)
    k80 = out["variant_source"] == "k80"
    maximum = out.loc[k80].groupby("anchor_id")["n_mut"].transform("max")
    if (maximum <= 0).any():
        raise ValueError("each K80 anchor must contain a non-zero-distance variant")
    out["distance_scale"] = 1.0
    out.loc[k80, "distance_scale"] = (out.loc[k80, "n_mut"] / maximum) ** float(alpha)

    out["training_label"] = out["pred_score"].astype(float)
    delta = out["pred_score"] - out["anchor_teacher"]
    out.loc[k80, "training_label"] = out.loc[k80, "anchor_measured"] + delta.loc[k80] * out.loc[k80, "distance_scale"]
    wild = out["variant_source"] == "wt"
    out.loc[wild, "training_label"] = out.loc[wild, "mean_te"]
    out["label_rule"] = out["variant_source"].map({
        "wt": "measured_wt", "k80": "wt_anchored_tdc",
        "flow_matching": "absolute_teacher", "matched_random": "absolute_teacher",
    })
    return out.drop(columns=["anchor_sequence"])


def assert_no_held_out_anchors(frame: pd.DataFrame, held_out: pd.DataFrame) -> None:
    if "transcript_id" not in held_out:
        raise ValueError("held-out table lacks 'transcript_id'")
    overlap = set(frame["transcript_id"].map(_base_id)) & set(held_out["transcript_id"].map(_base_id))
    if overlap:
        raise ValueError(f"leakage: {len(overlap)} held-out anchors occur in label input")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--held-out", action="append", default=[], help="Val/test table; repeatable and mandatory")
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    if not args.held_out:
        parser.error("at least one --held-out validation/test table is required")
    frame = _read(Path(args.input))
    for path in args.held_out:
        assert_no_held_out_anchors(frame, _read(Path(path)))
    result = construct_labels(frame, alpha=args.alpha)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix in {".parquet", ".pq"}:
        result.to_parquet(output, index=False)
    else:
        result.to_csv(output, index=False)
    print(result.groupby(["variant_source", "label_rule"]).size().to_string())


if __name__ == "__main__":
    main()
