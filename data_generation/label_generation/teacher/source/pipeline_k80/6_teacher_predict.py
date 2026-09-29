"""
4_teacher_predict.py — LightGBM teacher prediction + 3 soft labels
Loads embedding shards one by one to avoid OOM.
"""
import os, gc, time, pickle, yaml, sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
cfg_path = os.environ.get("ABLATION_CONFIG", os.path.join(ROOT_DIR, "config.yaml"))
with open(cfg_path) as f:
    CFG = yaml.safe_load(f)

OUTPUT_DIR = os.path.join(ROOT_DIR, CFG["output_dir"], "k80")
FEATURES_FILE = os.path.join(OUTPUT_DIR, "features.parquet")
SHARD_DIR = os.path.join(OUTPUT_DIR, "emb_shards")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "student_train.parquet")

ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
TEACHER_MODEL = os.path.join(ASSETS_DIR, "teacher_model.pkl")
HL_HUMAN = os.path.join(ASSETS_DIR, "human_time.csv")
HL_MOUSE = os.path.join(ASSETS_DIR, "mouse_time.csv")

N_WORKERS = CFG["n_workers"]
ALPHA = CFG["alpha"]

DRY_RUN = "--dry-run" in sys.argv


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


def compute_n_mut_parallel(df):
    print("Computing n_mut (hamming distance)...")
    t0 = time.time()
    df['clean_tid'] = df['transcript_id'].str.replace(r'_mut\d+$', '', regex=True)
    wt_mask = df['mut_step'] == 0
    wt_mrna = dict(zip(df.loc[wt_mask, 'clean_tid'], df.loc[wt_mask, 'mrna']))
    df['wt_mrna'] = df['clean_tid'].map(wt_mrna)

    chunk_size = max(len(df) // (N_WORKERS * 4), 1000)
    tasks = []
    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        tasks.append((start, df['mrna'].iloc[start:end].tolist(),
                       df['wt_mrna'].iloc[start:end].tolist()))

    n_mut = np.zeros(len(df), dtype=np.int32)
    with Pool(N_WORKERS) as pool:
        for start_idx, result in tqdm(pool.imap_unordered(hamming_chunk, tasks),
                                       total=len(tasks), desc="  Hamming"):
            n_mut[start_idx:start_idx + len(result)] = result

    df['n_mut'] = n_mut
    df.drop(columns=['wt_mrna'], inplace=True)
    print(f"  Done in {time.time()-t0:.1f}s")
    return df


def load_embedding_shards(shard_dir):
    """Load shards one by one, build tid->row_index mapping and a single numpy matrix."""
    import glob
    shard_files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pkl")))
    print(f"  Loading {len(shard_files)} embedding shards from {shard_dir}...")

    all_keys = []
    all_embs = []
    for sf in tqdm(shard_files, desc="  Loading shards"):
        with open(sf, "rb") as f:
            d = pickle.load(f)
        all_keys.extend(d["keys"])
        all_embs.append(d["embeddings"])
        del d; gc.collect()

    emb_matrix = np.concatenate(all_embs, axis=0)
    del all_embs; gc.collect()

    key_to_idx = {k: i for i, k in enumerate(all_keys)}
    print(f"  Total embeddings: {len(key_to_idx):,}, dim: {emb_matrix.shape[1]}")
    return key_to_idx, emb_matrix


def main():
    t0 = time.time()
    print("=" * 60)
    print("Step 4: Teacher Predict + Soft Labels")
    print("=" * 60)

    print(f"\nLoading features...")
    df = pd.read_parquet(FEATURES_FILE)
    if DRY_RUN:
        df = df.head(640)
    print(f"  Rows: {len(df):,}, Cols: {len(df.columns)}")

    # Load embeddings from shards
    key_to_idx, emb_matrix = load_embedding_shards(SHARD_DIR)
    emb_dim = emb_matrix.shape[1]

    missing = set(df['transcript_id']) - set(key_to_idx.keys())
    if missing:
        print(f"  WARNING: {len(missing)} missing embeddings, dropping")
        df = df[df['transcript_id'].isin(key_to_idx)].reset_index(drop=True)

    print("  Aligning embeddings to feature dataframe...")
    row_indices = np.array([key_to_idx[tid] for tid in df['transcript_id']])
    emb_aligned = emb_matrix[row_indices]
    del emb_matrix, key_to_idx, row_indices; gc.collect()

    feat_cols = [f"feat_{j}" for j in range(emb_dim)]
    df_emb = pd.DataFrame(emb_aligned, columns=feat_cols, index=df.index)
    df = pd.concat([df, df_emb], axis=1)
    del emb_aligned, df_emb; gc.collect()

    # Half-life mapping
    print("Mapping half-life...")
    for species_name, hl_path in [('human', HL_HUMAN), ('mouse', HL_MOUSE)]:
        if os.path.exists(hl_path):
            df_time = pd.read_csv(hl_path)
            hl_map = df_time.drop_duplicates('Ensembl Gene Id').set_index('Ensembl Gene Id')['half-life (PC1)']
            sp_mask = df['species'] == species_name
            wt_mask = sp_mask & (df['mut_step'] == 0)
            clean_genes = df.loc[wt_mask, 'gene_id'].astype(str).str.split('.').str[0]
            mapped_hl = clean_genes.map(hl_map)
            if 'hl' not in df.columns:
                df['hl'] = np.nan
            df.loc[wt_mask, 'hl'] = mapped_hl.values
    for sp in df['species'].unique():
        sp_mask = df['species'] == sp
        median_hl = df.loc[sp_mask & (df['mut_step'] == 0), 'hl'].median()
        df.loc[sp_mask & df['hl'].isna(), 'hl'] = median_hl

    # Teacher prediction
    print("Loading teacher model...")
    import joblib
    model = joblib.load(TEACHER_MODEL)
    feat_names = model.feature_name() if hasattr(model, 'feature_name') else model.booster_.feature_name()
    print(f"  Teacher features: {len(feat_names)}")

    available = set(df.columns)
    missing_feats = [f for f in feat_names if f not in available]
    if missing_feats:
        print(f"  WARNING: {len(missing_feats)} features missing, filling with 0")
        for f in missing_feats:
            df[f] = 0.0

    print("Running teacher prediction...")
    X = df[feat_names].values.astype(np.float32)
    os.environ['OMP_NUM_THREADS'] = str(min(80, N_WORKERS))
    try:
        pred = model.predict(X, num_threads=min(80, N_WORKERS))
    except:
        pred = model.predict(X)
    df['pred_score'] = pred
    del X; gc.collect()

    # Anchors
    df['clean_tid'] = df['transcript_id'].str.replace(r'_mut\d+$', '', regex=True)
    wt = df[df['mut_step'] == 0]
    d_pred_wt = dict(zip(wt['clean_tid'], wt['pred_score']))
    d_real_wt = dict(zip(wt['clean_tid'], wt['mean_te']))
    df['anchor_pred_wt'] = df['clean_tid'].map(d_pred_wt)
    df['anchor_real_wt'] = df['clean_tid'].map(d_real_wt)

    # Soft labels
    df['soft_label_org'] = np.where(df['mut_step'] == 0, df['anchor_real_wt'], df['pred_score'])
    delta = df['pred_score'] - df['anchor_pred_wt']
    df['soft_label_taylor'] = df['anchor_real_wt'] + delta

    df = compute_n_mut_parallel(df)
    max_mut = df.groupby('clean_tid')['n_mut'].transform('max').astype(np.float64)
    max_mut_safe = np.where(max_mut > 0, max_mut, 1.0)
    ratio = (df['n_mut'].values.astype(np.float64) / max_mut_safe) ** ALPHA
    df['soft_label_taylor_distance_correction'] = df['anchor_real_wt'].values + delta.values * ratio

    # QC
    wt_check = df[df['mut_step'] == 0]
    print(f"  WT org MAE: {(wt_check['soft_label_org'] - wt_check['mean_te']).abs().mean():.8f}")
    print(f"  WT taylor MAE: {(wt_check['soft_label_taylor'] - wt_check['mean_te']).abs().mean():.8f}")

    # Save
    out_cols = ['transcript_id', 'gene_id', 'species', 'mut_step',
                'utr5_size', 'cds_size', 'utr3_size', 'mrna', 'mean_te',
                'pred_score', 'anchor_pred_wt', 'anchor_real_wt',
                'soft_label_org', 'soft_label_taylor',
                'soft_label_taylor_distance_correction', 'n_mut']
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_parquet(OUTPUT_FILE, index=False)
    print(f"  Rows: {len(df):,}, Saved: {OUTPUT_FILE}")
    print(f"  Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
