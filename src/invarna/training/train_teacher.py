"""Fit the recorded privileged LightGBM TE teacher."""

import argparse
from pathlib import Path
import joblib
import pandas as pd
import yaml
from lightgbm import LGBMRegressor, early_stopping


def read_table(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix in {".parquet", ".pq"} else pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="data_generation/label_generation/configs/te_teacher.yaml")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    train, validation = read_table(args.train), read_table(args.validation)
    label = config["label_col"]
    excluded = set(config["non_feature_columns"] + [label])
    features = [column for column in train.columns if column not in excluded]
    if len(features) != config["expected_feature_count"]:
        raise ValueError(
            f"expected {config['expected_feature_count']} teacher features, found {len(features)}; "
            "missing features are never synthesized"
        )
    missing = set(features + [label]) - set(validation.columns)
    if missing:
        raise ValueError(f"validation table missing columns: {sorted(missing)}")
    params = dict(config["model"])
    params.pop("type", None)
    params.pop("best_iteration", None)
    stopping_rounds = params.pop("early_stopping_rounds")
    model = LGBMRegressor(**params)
    model.fit(
        train[features], train[label],
        eval_set=[(validation[features], validation[label])],
        callbacks=[early_stopping(stopping_rounds=stopping_rounds)],
    )
    output = Path(args.output or config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    print(f"saved {output}; features={len(features)} best_iteration={model.best_iteration_}")


if __name__ == "__main__":
    main()
