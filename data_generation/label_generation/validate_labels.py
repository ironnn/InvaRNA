"""
check_labels.py — Post-hoc validation of soft labels vs WT mean_te
For K80 and matched outputs, at different mut_id checkpoints,
compute R²/Pearson/Spearman of each soft label strategy against WT's mean_te.
Also generates comparison plots.
"""
import os, sys, yaml
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
cfg_path = os.environ.get(
    "ABLATION_CONFIG", os.path.join(ROOT_DIR, "configs", "student", "tdc_final.yaml")
)
with open(cfg_path) as f:
    CFG = yaml.safe_load(f)

OUT_BASE = CFG["output_dir"]
K80_FILE = os.path.join(ROOT_DIR, OUT_BASE, "k80", "train_lite_k80.parquet")
MATCHED_FILE = os.path.join(ROOT_DIR, OUT_BASE, "matched", "train_lite_matched.parquet")
PLOT_DIR = os.path.join(ROOT_DIR, "compare")

LABEL_COLS = {
    "pred_score": "Teacher Raw",
    "soft_label_org": "Org",
    "soft_label_taylor": "Taylor",
    "soft_label_taylor_dc": "Taylor DC",
    "soft_label_decoupled": "Decoupled",
}
CHECKPOINTS = [0, 1, 50, 100, 150, 200, 215]
COLORS = {
    "pred_score": "#9CA3AF",
    "soft_label_org": "#EF4444",
    "soft_label_taylor": "#F59E0B",
    "soft_label_taylor_dc": "#10B981",
    "soft_label_decoupled": "#3B82F6",
}


def evaluate(df, name):
    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"{'='*80}")

    df["mut_id"] = df["transcript_id"].str.extract(r"_mut(\d+)$")[0].astype(int)
    df["base_id"] = df["transcript_id"].str.rsplit("_", n=1).str[0]

    # WT mean_te lookup
    wt = df[df["mut_id"] == 0].set_index("base_id")["mean_te"].to_dict()

    # Available label cols
    avail_cols = [c for c in LABEL_COLS if c in df.columns]

    # --- Table: R² ---
    header = f"{'mut_id':>8s}  {'count':>6s}"
    for col in avail_cols:
        header += f"  {LABEL_COLS[col]:>12s}"
    print(header)
    print("-" * len(header))

    metrics_all = {}  # {mut_id: {col: {r2, mae, pearson, spearman}}}

    for ckpt in CHECKPOINTS:
        sub = df[df["mut_id"] == ckpt].copy()
        if len(sub) == 0:
            print(f"{ckpt:>8d}  {'--':>6s}")
            continue
        sub["wt_te"] = sub["base_id"].map(wt)
        sub = sub.dropna(subset=["wt_te"])
        if len(sub) < 10:
            print(f"{ckpt:>8d}  {len(sub):>6d}  (too few)")
            continue

        row = f"{ckpt:>8d}  {len(sub):>6d}"
        metrics_all[ckpt] = {}
        for col in avail_cols:
            if col in sub.columns and sub[col].notna().sum() > 10:
                y_true = sub["wt_te"].values
                y_pred = sub[col].values
                r2 = r2_score(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                pr, _ = pearsonr(y_true, y_pred)
                sr, _ = spearmanr(y_true, y_pred)
                metrics_all[ckpt][col] = {"r2": r2, "mae": mae, "pearson": pr, "spearman": sr}
                row += f"  {r2:>12.4f}"
            else:
                row += f"  {'N/A':>12s}"
        print(row)

    # WT sanity check
    wt_df = df[df["mut_id"] == 0].copy()
    if len(wt_df) > 0 and "pred_score" in wt_df.columns:
        r2_wt = r2_score(wt_df["mean_te"], wt_df["pred_score"])
        print(f"\n  WT sanity check: pred_score vs mean_te  R²={r2_wt:.4f}")

    # --- Detailed metrics table ---
    print(f"\n  Detailed metrics (MAE / Pearson / Spearman):")
    for ckpt in CHECKPOINTS:
        if ckpt not in metrics_all:
            continue
        print(f"\n  mut_id={ckpt}:")
        for col in avail_cols:
            if col in metrics_all[ckpt]:
                m = metrics_all[ckpt][col]
                print(f"    {LABEL_COLS[col]:>12s}:  R²={m['r2']:.4f}  MAE={m['mae']:.4f}  Pearson={m['pearson']:.4f}  Spearman={m['spearman']:.4f}")

    # --- Plots ---
    plot_scatter(df, wt, avail_cols, name, metrics_all)
    plot_distribution(df, wt, avail_cols, name)
    plot_r2_trend(metrics_all, avail_cols, name)

    return metrics_all


def plot_scatter(df, wt_map, avail_cols, name, metrics_all):
    """Scatter plots: soft label vs WT mean_te at each checkpoint."""
    plot_ckpts = [c for c in CHECKPOINTS if c in metrics_all and c > 0]
    if not plot_ckpts:
        return

    n_cols = len(avail_cols)
    n_rows = len(plot_ckpts)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)

    for i, ckpt in enumerate(plot_ckpts):
        sub = df[df["mut_id"] == ckpt].copy()
        sub["wt_te"] = sub["base_id"].map(wt_map)
        sub = sub.dropna(subset=["wt_te"])

        for j, col in enumerate(avail_cols):
            ax = axes[i][j]
            if col not in sub.columns or sub[col].isna().all():
                ax.set_visible(False)
                continue
            x = sub["wt_te"].values
            y = sub[col].values
            ax.scatter(x, y, s=8, alpha=0.4, color=COLORS.get(col, "#666"))
            # diagonal
            lims = [min(x.min(), y.min()), max(x.max(), y.max())]
            ax.plot(lims, lims, 'k--', alpha=0.3, lw=1)
            m = metrics_all.get(ckpt, {}).get(col, {})
            r2 = m.get("r2", float("nan"))
            ax.set_title(f"mut{ckpt} | {LABEL_COLS[col]} | R²={r2:.3f}", fontsize=9)
            ax.set_xlabel("WT mean_te", fontsize=8)
            ax.set_ylabel(LABEL_COLS[col], fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(f"{name}: Soft Label vs WT mean_te", fontsize=12, y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"label_scatter_{name.lower()}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_distribution(df, wt_map, avail_cols, name):
    """Distribution plots: KDE overlay of soft label distributions at key checkpoints."""
    plot_ckpts = [c for c in [1, 50, 100, 200] if c in df["mut_id"].values]
    if not plot_ckpts:
        return

    fig, axes = plt.subplots(1, len(plot_ckpts), figsize=(5 * len(plot_ckpts), 4), squeeze=False)

    for i, ckpt in enumerate(plot_ckpts):
        ax = axes[0][i]
        sub = df[df["mut_id"] == ckpt].copy()
        sub["wt_te"] = sub["base_id"].map(wt_map)
        sub = sub.dropna(subset=["wt_te"])

        # KDE for WT
        wt_vals = sub["wt_te"].dropna()
        if len(wt_vals) > 5:
            wt_vals.plot.kde(ax=ax, color="black", linewidth=2, label="WT mean_te", alpha=0.8)

        for col in avail_cols:
            if col in sub.columns and sub[col].notna().sum() > 5:
                sub[col].plot.kde(ax=ax, color=COLORS.get(col, "#666"),
                                 linewidth=1.5, label=LABEL_COLS[col], alpha=0.7)

        ax.set_title(f"mut{ckpt}", fontsize=10)
        ax.set_xlabel("value", fontsize=9)
        ax.set_xlim(wt_vals.quantile(0.01) - 0.5, wt_vals.quantile(0.99) + 0.5)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(f"{name}: Label Distributions (KDE)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"label_dist_{name.lower()}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_r2_trend(metrics_all, avail_cols, name):
    """R² trend across mut_id checkpoints."""
    ckpts = sorted([c for c in metrics_all if c > 0])
    if not ckpts:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for col in avail_cols:
        r2s = [metrics_all[c].get(col, {}).get("r2", float("nan")) for c in ckpts]
        if all(np.isnan(r2s)):
            continue
        ax.plot(ckpts, r2s, marker="o", label=LABEL_COLS[col], color=COLORS.get(col, "#666"), linewidth=2)

    ax.set_xlabel("mut_id", fontsize=11)
    ax.set_ylabel("R² vs WT mean_te", fontsize=11)
    ax.set_title(f"{name}: R² Trend by Mutation Distance", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.05)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"label_r2_trend_{name.lower()}.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("=" * 80)
    print("  Soft Label Validation (R² vs WT mean_te + plots)")
    print("=" * 80)

    for label, path in [("K80", K80_FILE), ("Matched", MATCHED_FILE)]:
        if not os.path.exists(path):
            print(f"\n  SKIP {label}: {path} not found")
            continue
        df = pd.read_parquet(path)
        evaluate(df, label)

    print(f"\n{'='*80}")
    print("  Done.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
