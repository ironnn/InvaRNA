"""Loading and scoring helpers for the bundled LightGBM TE teacher."""

from pathlib import Path

import joblib


def load_teacher(path="assets/checkpoints/teacher/privileged_teacher.pkl"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
    return joblib.load(path)


def predict_teacher(features, path="assets/checkpoints/teacher/privileged_teacher.pkl"):
    return load_teacher(path).predict(features)
