#!/usr/bin/env python
"""Fit the exact Fig. 2B LightGBM probe on cached MRL embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgbm
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


PARAMETERS = {
    "objective": "regression",
    "learning_rate": 0.01,
    "n_estimators": 1200,
    "num_leaves": 40,
    "max_depth": 11,
    "min_child_samples": 26,
    "subsample": 0.82,
    "colsample_bytree": 0.3,
    "reg_alpha": 0.2,
    "reg_lambda": 0.8,
    "n_jobs": -1,
    "verbose": -1,
}


def load_split(directory: Path, model: str, split: str):
    features = np.load(directory / f"{model}_{split}.npy")
    labels = np.load(directory / f"{model}_{split}.labels.npy")
    return features, labels


def metrics(labels, predictions):
    return {
        "r2": float(r2_score(labels, predictions)),
        "spearman": float(spearmanr(labels, predictions)[0]),
        "pearson": float(pearsonr(labels, predictions)[0]),
        "mae": float(mean_absolute_error(labels, predictions)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emb-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_x, train_y = load_split(args.emb_dir, args.model, "train")
    validation_x, validation_y = load_split(args.emb_dir, args.model, "val")
    test_x, test_y = load_split(args.emb_dir, args.model, "test")
    model = lgbm.LGBMRegressor(**PARAMETERS)
    model.fit(
        train_x,
        train_y,
        eval_set=[(validation_x, validation_y)],
        eval_metric="rmse",
        callbacks=[lgbm.early_stopping(50, verbose=False)],
    )
    result = {
        "best_iteration": int(model.best_iteration_),
        "model": args.model,
        "parameters": PARAMETERS,
        "validation": metrics(validation_y, model.predict(validation_x)),
        "test": metrics(test_y, model.predict(test_x)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
