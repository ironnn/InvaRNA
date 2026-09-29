"""Per-species regression and rank-correlation summaries."""

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score


def summarize(frame: pd.DataFrame, truth: str, prediction: str, species: str = "species"):
    rows = []
    for name, group in frame.dropna(subset=[truth, prediction]).groupby(species):
        rows.append({"species": name, "n": len(group), "r2": r2_score(group[truth], group[prediction]),
                     "pearson": pearsonr(group[truth], group[prediction]).statistic,
                     "spearman": spearmanr(group[truth], group[prediction]).statistic})
    return pd.DataFrame(rows)
