"""LightGBM probes for frozen representation-transfer benchmarks."""

from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score


def lightgbm_probe(train_x, train_y, validation_x, validation_y, test_x, test_y, *, params):
    """Fit an explicitly configured probe and report held-out test R2."""
    if not params:
        raise ValueError("author-confirmed LightGBM parameters are required")
    model = LGBMRegressor(**params)
    model.fit(train_x, train_y, eval_set=[(validation_x, validation_y)])
    predictions = model.predict(test_x)
    return {"model": model, "predictions": predictions, "r2": r2_score(test_y, predictions)}
