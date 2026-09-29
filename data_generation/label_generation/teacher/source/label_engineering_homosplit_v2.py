"""
label_engineering_homosplit_v2.py — Label engineering with orthology-aware leakage removal

Same as label_engineering.py, but additionally removes mouse orthologs of
val/test human genes (and all their mutants) from the training set.

Leakage removal logic:
  1. (original) Remove val/test human genes + their mutants
  2. (NEW v2)   Remove mouse orthologs of val/test human genes + their mutants
     Uses RBH (reciprocal best hit) pairs from assets/rbh_human_to_mouse.parquet

Reads from:  {output_dir}/{k80|matched}/student_train.parquet
Writes to:   {output_dir}/{k80|matched}/train_lite_{k80|matched}.parquet
"""
import os, sys, gc, time, yaml
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = SCRIPT_DIR
cfg_path = os.environ.get("ABLATION_CONFIG", os.path.join(ROOT_DIR, "config.yaml"))
with open(cfg_path) as f:
    CFG = yaml.safe_load(f)

N_WORKERS = CFG["n_workers"]
INVARNA = CFG["invarna_root"]
INVARNA_LOCAL = os.path.join(os.path.dirname(ROOT_DIR), "InvaRNA")
VAL_PATH = os.path.join(INVARNA_LOCAL, "data/train/val_lite.parquet")
TEST_PATH = os.path.join(INVARNA_LOCAL, "data/train/test_lite.parquet")
RBH_H2M = os.path.join(ROOT_DIR, "assets", "rbh_human_to_mouse.parquet")


def load_leak_ids():
    """Load val+test transcript base IDs to exclude from training data."""
    leak = set()
    for path in [VAL_PATH, TEST_PATH]:
        if os.path.exists(path):
            df = pd.read_parquet(path, columns=["transcript_id"])
            ids = df["transcript_id"].str.replace(r"_mut\d+$", "", regex=True)
            leak.update(ids)
    return leak


def load_ortholog_leak_ids(human_leak_ids):
    """Given human val/test base IDs, find their mouse orthologs via RBH."""
    if not os.path.exists(RBH_H2M):
        print(f"  WARNING: {RBH_H2M} not found, skipping orthology removal")
        return set()
    rbh = pd.read_parquet(RBH_H2M)
    h2m = dict(zip(rbh['tid'], rbh['homolog_tid']))
    ortho = set()
    for tid in human_leak_ids:
        if tid in h2m:
            ortho.add(h2m[tid])
    return ortho


# Label engineering hyperparams
LE = CFG.get("label_engineering", {})
ALPHA       = float(LE.get("alpha", CFG.get("alpha", 1.0)))
C_MIN       = float(LE.get("c_min", 0.4))
GAMMA       = float(LE.get("gamma", 3.0))
LAMBDA_BASE = float(LE.get("lambda_base", 0.2))
BETA        = float(LE.get("beta", 3.0))

OUTLIER_MODE       = LE.get("outlier_mode", "drop")
DELTA_Q_HIGH       = float(LE.get("delta_q_high", 0.995))
LABEL_Q_LOW        = float(LE.get("label_q_low", 0.005))
LABEL_Q_HIGH       = float(LE.get("label_q_high", 0.995))
LABEL_MARGIN_RATIO = float(LE.get("label_margin_ratio", 0.1))

LITE_COLS = [
    'transcript_id', 'utr5_size', 'cds_size', 'utr3_size',
    'mrna', 'pred_score', 'data_type', 'species', 'gene_id', 'mean_te',
    'soft_label_org', 'soft_label_taylor',
    'soft_label_taylor_dc', 'soft_label_decoupled',
]


def hamming_chunk(args):
    idx, mut_seqs, wt_seqs = args
    results = np.zeros(len(mut_seqs), dtype=np.int32)
    for i in range(len(mut_seqs)):
        m, w = mut_seqs[i], wt_seqs[i]
        if len(m) == len(w):
            results[i] = sum(a != b for a, b in zip(m, w))
        else:
            results[i] = abs(len(m) - len(w))
    return idx, results


def compute_n_mut(df):
    print("  Computing hamming distances...")
    t0 = time.time()
    df['clean_tid'] = df['transcript_id'].str.replace(r'_mut\d+$', '', regex=True)
    wt_mask = df['mut_step'] == 0
    wt_mrna = dict(zip(df.loc[wt_mask, 'clean_tid'], df.loc[wt_mask, 'mrna']))
    df['wt_mrna'] = df['clean_tid'].map(wt_mrna)

    chunk_size = max(len(df) // (N_WORKERS * 4), 1000)
    tasks = []
    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        tasks.append((start, df['mrna'].values[start:end], df['wt_mrna'].values[start:end]))

    n_mut = np.zeros(len(df), dtype=np.int32)
    with Pool(N_WORKERS) as pool:
        for start_idx, result in tqdm(pool.imap_unordered(hamming_chunk, tasks),
                                       total=len(tasks), desc="    Hamming"):
            n_mut[start_idx:start_idx + len(result)] = result

    df['n_mut'] = n_mut
    df.drop(columns=['wt_mrna'], inplace=True)
    print(f"    Done in {time.time()-t0:.1f}s")
    return df


def filter_outliers(df):
    if OUTLIER_MODE == "none":
        df['soft_label_decoupled'] = df['soft_label_decoupled_raw']
        print(f"  Outlier filter: none (all rows kept)")
        return df

    is_wt = (df['mut_step'] == 0).values
    mut_mask = ~is_wt

    bad_numeric = (~np.isfinite(df['soft_label_decoupled_raw'].values)) | \
                  (~np.isfinite(df['teacher_delta'].values))

    delta_abs = df.loc[mut_mask, 'teacher_delta'].abs()
    delta_hi = delta_abs.quantile(DELTA_Q_HIGH)
    bad_delta = df['teacher_delta'].abs().values > delta_hi

    wt_labels = df.loc[is_wt, 'mean_te']
    low = wt_labels.quantile(LABEL_Q_LOW)
    high = wt_labels.quantile(LABEL_Q_HIGH)
    margin = LABEL_MARGIN_RATIO * (high - low)
    low_clip = low - margin
    high_clip = high + margin
    bad_range = (df['soft_label_decoupled_raw'].values < low_clip) | \
                (df['soft_label_decoupled_raw'].values > high_clip)

    flagged = mut_mask & (bad_numeric | bad_delta | bad_range)
    n_flagged = flagged.sum()
    n_mut = mut_mask.sum()

    print(f"  Outlier filter: {OUTLIER_MODE}")
    print(f"    delta_hi threshold (q={DELTA_Q_HIGH}): {delta_hi:.4f}")
    print(f"    label range: [{low_clip:.4f}, {high_clip:.4f}]")
    print(f"    bad_numeric: {(mut_mask & bad_numeric).sum()}")
    print(f"    bad_delta:   {(mut_mask & bad_delta).sum()}")
    print(f"    bad_range:   {(mut_mask & bad_range).sum()}")
    print(f"    total flagged: {n_flagged}/{n_mut} mutants ({100*n_flagged/max(n_mut,1):.2f}%)")

    if OUTLIER_MODE == "drop":
        df = df.loc[~flagged].copy()
        df['soft_label_decoupled'] = df['soft_label_decoupled_raw']
        print(f"    dropped {n_flagged} rows, remaining: {len(df):,}")

    elif OUTLIER_MODE == "clip":
        numeric_bad = mut_mask & bad_numeric
        df.loc[numeric_bad, 'soft_label_decoupled_raw'] = df.loc[numeric_bad, 'anchor_real_wt']
        df['soft_label_decoupled'] = df['soft_label_decoupled_raw'].clip(low_clip, high_clip)
        df.loc[is_wt, 'soft_label_decoupled'] = df.loc[is_wt, 'anchor_real_wt']
        print(f"    clipped {n_flagged} mutants to [{low_clip:.4f}, {high_clip:.4f}]")

    return df


def compute_labels(df):
    print("  Computing soft labels...")

    df['clean_tid'] = df['transcript_id'].str.replace(r'_mut\d+$', '', regex=True)
    wt = df[df['mut_step'] == 0]
    d_pred_wt = dict(zip(wt['clean_tid'], wt['pred_score']))
    d_real_wt = dict(zip(wt['clean_tid'], wt['mean_te']))
    df['anchor_pred_wt'] = df['clean_tid'].map(d_pred_wt)
    df['anchor_real_wt'] = df['clean_tid'].map(d_real_wt)

    y_w = df['anchor_real_wt'].values
    T_w = df['anchor_pred_wt'].values
    T_m = df['pred_score'].values
    delta = T_m - T_w
    df['teacher_delta'] = delta
    is_wt = (df['mut_step'] == 0).values

    df['soft_label_org'] = np.where(is_wt, y_w, T_m)
    df['soft_label_taylor'] = y_w + delta

    df = compute_n_mut(df)
    n_mut = df['n_mut'].values.astype(np.float64)
    max_mut = df.groupby('clean_tid')['n_mut'].transform('max').values.astype(np.float64)
    max_mut_safe = np.where(max_mut > 0, max_mut, 1.0)
    ratio = (n_mut / max_mut_safe) ** ALPHA
    df['soft_label_taylor_dc'] = y_w + delta * ratio

    epsilon_w = np.abs(T_w - y_w)
    c_w = np.maximum(C_MIN, np.exp(-GAMMA * epsilon_w))
    d_m = n_mut / max_mut_safe
    g_d = LAMBDA_BASE + (1.0 - LAMBDA_BASE) * np.exp(-BETA * d_m)
    df['soft_label_decoupled_raw'] = y_w + c_w * g_d * delta

    for col in ['soft_label_org', 'soft_label_taylor', 'soft_label_taylor_dc', 'soft_label_decoupled_raw']:
        df.loc[is_wt, col] = df.loc[is_wt, 'anchor_real_wt']

    df = filter_outliers(df)

    wt_check = df[df['mut_step'] == 0]
    for col in ['soft_label_org', 'soft_label_taylor', 'soft_label_taylor_dc', 'soft_label_decoupled']:
        if col in df.columns:
            mae = (wt_check[col] - wt_check['mean_te']).abs().mean()
            print(f"    WT {col} MAE: {mae:.8f}")

    mut_mask = df['mut_step'] != 0
    print(f"\n  Decoupled stats (mutants only):")
    if 'teacher_delta' in df.columns:
        c_w_vals = np.maximum(C_MIN, np.exp(-GAMMA * np.abs(df.loc[mut_mask, 'anchor_pred_wt'] - df.loc[mut_mask, 'anchor_real_wt'])))
        n_m = df.loc[mut_mask, 'n_mut'].values.astype(np.float64)
        max_m = df.loc[mut_mask].groupby('clean_tid')['n_mut'].transform('max').values.astype(np.float64)
        max_m_safe = np.where(max_m > 0, max_m, 1.0)
        g_d_vals = LAMBDA_BASE + (1.0 - LAMBDA_BASE) * np.exp(-BETA * n_m / max_m_safe)
        print(f"    c_w:  mean={c_w_vals.mean():.4f}  min={c_w_vals.min():.4f}  max={c_w_vals.max():.4f}")
        print(f"    g(d): mean={g_d_vals.mean():.4f}  min={g_d_vals.min():.4f}  max={g_d_vals.max():.4f}")
        print(f"    c*g:  mean={(c_w_vals*g_d_vals).mean():.4f}")

    return df


def process_pipeline(name, input_file, output_file):
    if not os.path.exists(input_file):
        print(f"\n  SKIP {name}: {input_file} not found")
        return

    print(f"\n{'='*60}")
    print(f"  Label Engineering (homosplit_v2): {name}")
    print(f"{'='*60}")

    df = pd.read_parquet(input_file)
    print(f"  Loaded: {len(df):,} rows")

    df = compute_labels(df)

    # --- Leakage removal (v2: original + orthology) ---
    human_leak_ids = load_leak_ids()
    ortho_leak_ids = load_ortholog_leak_ids(human_leak_ids)
    all_leak_ids = human_leak_ids | ortho_leak_ids

    print(f"\n  Leakage IDs: {len(human_leak_ids)} human val/test + {len(ortho_leak_ids)} mouse orthologs = {len(all_leak_ids)} total")

    if all_leak_ids:
        df['base_id'] = df['transcript_id'].str.replace(r'_mut\d+$', '', regex=True)
        leak_mask = df['base_id'].isin(all_leak_ids)
        n_leak = leak_mask.sum()
        if n_leak > 0:
            n_human = (leak_mask & (df['species'] == 'human')).sum()
            n_mouse = (leak_mask & (df['species'] == 'mouse')).sum()
            n_wt = (leak_mask & (df['mut_step'] == 0)).sum()
            n_mut = n_leak - n_wt
            print(f"  Removing: {n_leak:,} rows ({n_wt} WT + {n_mut:,} mutants)")
            print(f"    human: {n_human:,}, mouse: {n_mouse:,}")
            df = df[~leak_mask].copy()
            print(f"  After removal: {len(df):,} rows")
        else:
            print(f"  No overlap found. OK.")
        df.drop(columns=['base_id'], inplace=True, errors='ignore')

    df['data_type'] = 'train'
    out_cols = [c for c in LITE_COLS if c in df.columns]
    df_out = df[out_cols].copy()

    wt = df_out[df_out['transcript_id'].str.endswith('_mut0')]
    mut = df_out[~df_out['transcript_id'].str.endswith('_mut0')]
    print(f"\n  Final: WT={len(wt):,}, Mut={len(mut):,}, Total={len(df_out):,}")

    for col in ['soft_label_org', 'soft_label_taylor', 'soft_label_taylor_dc', 'soft_label_decoupled']:
        if col in df_out.columns:
            print(f"  {col} NaN: {df_out[col].isna().sum()}")

    df_out.to_parquet(output_file, index=False)
    print(f"  Saved: {output_file}")
    print(f"  Size: {os.path.getsize(output_file)/(1024**3):.2f} GB")


def main():
    t0 = time.time()
    print("=" * 60)
    print("  Label Engineering — homosplit_v2 (orthology-aware)")
    print(f"  alpha={ALPHA}, c_min={C_MIN}, gamma={GAMMA}")
    print(f"  lambda_base={LAMBDA_BASE}, beta={BETA}")
    print(f"  outlier_mode={OUTLIER_MODE}")
    print(f"  RBH file: {RBH_H2M}")
    print("=" * 60)

    out_base = os.path.join(ROOT_DIR, CFG["output_dir"])

    process_pipeline(
        "K80",
        os.path.join(out_base, "k80", "student_train.parquet"),
        os.path.join(out_base, "k80", "train_lite_k80_homo.parquet"),
    )

    process_pipeline(
        "Matched",
        os.path.join(out_base, "matched", "student_train.parquet"),
        os.path.join(out_base, "matched", "train_lite_matched_homo.parquet"),
    )

    print(f"\n{'='*60}")
    print(f"  Label Engineering (homosplit_v2) DONE in {time.time()-t0:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
