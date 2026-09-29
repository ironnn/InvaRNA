"""Attach bundled-teacher predictions to an engineered feature table."""

import argparse
import pandas as pd

from invarna.models.teacher import load_teacher


def score_feature_table(features: pd.DataFrame, model_path, id_columns=()) -> pd.DataFrame:
    model = load_teacher(model_path)
    if hasattr(model, "feature_name"):
        feature_columns = list(model.feature_name())
    elif hasattr(model, "booster_"):
        feature_columns = list(model.booster_.feature_name())
    else:
        raise TypeError("teacher does not expose its fitted feature names")
    missing = [column for column in feature_columns if column not in features]
    if missing:
        raise ValueError(
            f"feature table is missing {len(missing)} fitted teacher features; "
            "missing values are not synthesized"
        )
    result = features.copy()
    result["pred_score"] = model.predict(result[feature_columns])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Input parquet feature table")
    parser.add_argument("--model", default="assets/checkpoints/teacher/privileged_teacher.pkl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--id-columns", nargs="*", default=["transcript_id"])
    args = parser.parse_args()
    frame = pd.read_parquet(args.features)
    score_feature_table(frame, args.model, args.id_columns).to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
