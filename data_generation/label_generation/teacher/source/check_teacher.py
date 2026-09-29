"""
check_teacher.py — Validate teacher model R² on val/test sets
Uses multi-GPU parallel embedding extraction (same as pipeline).
"""
import os, sys, gc, time, pickle, yaml, glob, tempfile
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = SCRIPT_DIR
cfg_path = os.environ.get("ABLATION_CONFIG", os.path.join(ROOT_DIR, "config.yaml"))
with open(cfg_path) as f:
    CFG = yaml.safe_load(f)

INVARNA = CFG["invarna_root"]
sys.path.insert(0, INVARNA)
sys.path.insert(0, os.path.join(INVARNA, "src"))
sys.path.insert(0, os.path.join(ROOT_DIR, "utils"))

ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
TEACHER_MODEL = os.path.join(ASSETS_DIR, "teacher_model.pkl")
HL_HUMAN = os.path.join(ASSETS_DIR, "human_time.csv")
MODEL_WEIGHTS = os.path.join(INVARNA, "assets/checkpoints/backbone/pretrained_step13500.pt")
MODEL_CONFIG = os.path.join(INVARNA, "config/backbone.yaml")

GPU_IDS = CFG["gpu_ids"]
NUM_GPUS = len(GPU_IDS)
BATCH_SIZE = CFG["batch_size"]
TOTAL_LENGTH = 10000
FIXED_CDS_START = 1000

SPLITS = {
    "val": os.path.join(INVARNA, "data/train/val_lite.parquet"),
    "test": os.path.join(INVARNA, "data/train/test_lite.parquet"),
}


# --- Embedding (multi-GPU, same logic as pipeline_k80/4_extract_embedding.py) ---

class AlignmentDataset(Dataset):
    def __init__(self, df_path, tokenizer):
        self.tokenizer = tokenizer
        self.unk_id = tokenizer.convert_tokens_to_ids("N")
        self.pad_id = tokenizer.pad_token_id
        self.df = pd.read_pickle(df_path)

    def force_align_sequence(self, raw_seq, u5_len, cds_len, u3_len):
        seq = str(raw_seq).upper()
        u5_truncate_offset = max(0, u5_len - FIXED_CDS_START)
        if u5_truncate_offset > 0:
            seq = seq[u5_truncate_offset:]
            current_u5_len = FIXED_CDS_START
        else:
            current_u5_len = u5_len
        left_padding_count = FIXED_CDS_START - current_u5_len
        padded_seq = ('N' * left_padding_count) + seq
        padded_seq = padded_seq[:TOTAL_LENGTH].ljust(TOTAL_LENGTH, 'N')
        range_u5 = (left_padding_count, FIXED_CDS_START)
        range_cds = (FIXED_CDS_START, min(FIXED_CDS_START + cds_len, TOTAL_LENGTH))
        range_u3 = (min(FIXED_CDS_START + cds_len, TOTAL_LENGTH),
                     min(FIXED_CDS_START + cds_len + u3_len, TOTAL_LENGTH))
        return padded_seq, range_u5, range_cds, range_u3

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        padded_seq, u5_rng, cds_rng, u3_rng = self.force_align_sequence(
            row['mrna'], int(row['utr5_size']), int(row['cds_size']), int(row['utr3_size']))
        input_ids = self.tokenizer.encode(padded_seq, add_special_tokens=False)
        input_ids = [self.pad_id if t == self.unk_id else t for t in input_ids]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "gene_id": str(row['transcript_id']),
            "u5_rng": u5_rng, "cds_rng": cds_rng, "u3_rng": u3_rng,
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "gene_id": [item["gene_id"] for item in batch],
        "u5_rng": [item["u5_rng"] for item in batch],
        "cds_rng": [item["cds_rng"] for item in batch],
        "u3_rng": [item["u3_rng"] for item in batch],
    }


def gpu_worker(rank, gpu_id, chunk_path, shard_path):
    """Single GPU worker — same logic as pipeline."""
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{gpu_id}")

    from trainmodule import SequenceLightningModule
    from backbone.tokenization import InvaRNATokenizer

    config = OmegaConf.load(MODEL_CONFIG)
    model = SequenceLightningModule(config=config)
    state_dict = torch.load(MODEL_WEIGHTS, map_location="cpu")
    model.model.load_state_dict(state_dict)
    model.eval(); model.freeze()
    backbone = model.model.invaRNA
    backbone.config.return_dict = True
    model = model.to(device)

    tokenizer = InvaRNATokenizer(model_max_length=TOTAL_LENGTH)
    dataset = AlignmentDataset(chunk_path, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn, num_workers=1, pin_memory=True)

    keys, emb_list = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        with torch.no_grad():
            hidden = backbone(input_ids=input_ids)[0]
        for i, gid in enumerate(batch["gene_id"]):
            h = hidden[i]
            u5_s, u5_e = batch["u5_rng"][i]
            cds_s, cds_e = batch["cds_rng"][i]
            u3_s, u3_e = batch["u3_rng"][i]

            def pool(start, end):
                if start >= end:
                    return torch.zeros(h.size(-1) * 2, device=device)
                region = h[start:end]
                return torch.cat([region.mean(dim=0), region.max(dim=0)[0]])

            emb = torch.cat([pool(u5_s, u5_e), pool(cds_s, cds_e), pool(u3_s, u3_e)])
            keys.append(gid)
            emb_list.append(emb.cpu().numpy().astype(np.float32))

    emb_matrix = np.stack(emb_list) if emb_list else np.empty((0, 0), dtype=np.float32)
    with open(shard_path, "wb") as f:
        pickle.dump({"keys": keys, "embeddings": emb_matrix}, f)
    print(f"  [GPU {gpu_id}] {len(keys)} sequences → {emb_matrix.shape}")


def extract_embeddings_multi_gpu(df, tmp_dir):
    """Split df across GPUs, extract embeddings in parallel, merge."""
    n = len(df)
    num_gpus = min(NUM_GPUS, max(1, n // 10))  # don't use more GPUs than needed
    indices = np.array_split(range(n), num_gpus)

    chunk_paths, shard_paths = [], []
    for i, idx in enumerate(indices):
        chunk = df.iloc[idx]
        cp = os.path.join(tmp_dir, f"chunk_{i}.pkl")
        sp = os.path.join(tmp_dir, f"shard_{i}.pkl")
        chunk.to_pickle(cp)
        chunk_paths.append(cp)
        shard_paths.append(sp)

    processes = []
    for i in range(num_gpus):
        p = mp.Process(target=gpu_worker, args=(i, GPU_IDS[i], chunk_paths[i], shard_paths[i]))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # Merge shards
    all_keys, all_embs = [], []
    for sp in shard_paths:
        d = pickle.load(open(sp, "rb"))
        all_keys.extend(d["keys"])
        all_embs.append(d["embeddings"])
    emb_matrix = np.concatenate(all_embs, axis=0)

    # Cleanup
    for f in chunk_paths + shard_paths:
        os.remove(f)

    return all_keys, emb_matrix


def main():
    mp.set_start_method('spawn', force=True)
    t0 = time.time()
    print("=" * 60)
    print("  Teacher Model Validation (val + test)")
    print(f"  GPUs: {GPU_IDS[:NUM_GPUS]}")
    print("=" * 60)

    # Load teacher model
    import joblib
    model = joblib.load(TEACHER_MODEL)
    feat_names = model.feature_name_
    print(f"  Teacher features: {len(feat_names)}")

    # Half-life mapping
    hl_map = {}
    if os.path.exists(HL_HUMAN):
        df_hl = pd.read_csv(HL_HUMAN)
        hl_map = df_hl.drop_duplicates('Ensembl Gene Id').set_index('Ensembl Gene Id')['half-life (PC1)'].to_dict()

    from lgbm_feature_extract_from_str import fe, get_cols
    FEATURE_LIST = ['LL', 'P5', 'P3', 'CF', 'AAF', '3mer_freq_5', 'K', 'DC', 'Struct']
    feat_col_names = get_cols(FEATURE_LIST)

    tmp_dir = tempfile.mkdtemp(prefix="check_teacher_")

    results = {}
    for split_name, split_path in SPLITS.items():
        if not os.path.exists(split_path):
            print(f"\n  SKIP {split_name}: {split_path} not found")
            continue

        print(f"\n--- {split_name} ({split_path}) ---")
        df = pd.read_parquet(split_path)
        print(f"  Rows: {len(df)}")

        # 1. Handcrafted features
        print("  Extracting handcrafted features...")
        feat_rows = []
        for _, row in df.iterrows():
            feats = fe(FEATURE_LIST, row['mrna'], row['utr5_size'], row['cds_size'], row['utr3_size'], len(row['mrna']))
            feat_rows.append(feats)
        feat_df = pd.DataFrame(feat_rows, columns=feat_col_names, index=df.index)
        print(f"    {feat_df.shape[1]} features")

        # 2. Embeddings (multi-GPU)
        print(f"  Extracting embeddings ({min(NUM_GPUS, max(1, len(df)//10))} GPUs)...")
        keys, emb_matrix = extract_embeddings_multi_gpu(df, tmp_dir)
        print(f"    Embedding: {emb_matrix.shape}")

        # Align embedding order to df
        key_to_idx = {k: i for i, k in enumerate(keys)}
        row_indices = [key_to_idx[tid] for tid in df['transcript_id']]
        emb_matrix = emb_matrix[row_indices]

        # 3. Combine
        emb_cols = [f"feat_{j}" for j in range(emb_matrix.shape[1])]
        df_emb = pd.DataFrame(emb_matrix, columns=emb_cols, index=df.index)
        df_all = pd.concat([feat_df, df_emb], axis=1)

        # Half-life
        clean_genes = df['gene_id'].astype(str).str.split('.').str[0]
        df_all['hl'] = clean_genes.map(hl_map)
        df_all['hl'] = df_all['hl'].fillna(df_all['hl'].median())

        # Fill missing features
        missing = [f for f in feat_names if f not in df_all.columns]
        if missing:
            df_all = pd.concat([df_all, pd.DataFrame(0.0, index=df_all.index, columns=missing)], axis=1)

        # 4. Predict
        print("  Running teacher prediction...")
        X = df_all[feat_names].values.astype(np.float32)
        pred = model.predict(X)

        # 5. Metrics
        y_true = df['mean_te'].values
        r2 = r2_score(y_true, pred)
        mae = mean_absolute_error(y_true, pred)
        pr, _ = pearsonr(y_true, pred)
        sr, _ = spearmanr(y_true, pred)

        print(f"\n  {split_name} Results:")
        print(f"    R²={r2:.4f}  MAE={mae:.4f}  Pearson={pr:.4f}  Spearman={sr:.4f}")
        results[split_name] = {"R2": r2, "MAE": mae, "Pearson": pr, "Spearman": sr}

        del df_all, emb_matrix, feat_df, df_emb; gc.collect()

    # Cleanup tmp
    os.rmdir(tmp_dir)

    print(f"\n{'='*60}")
    print(f"  Teacher Validation Summary")
    print(f"{'='*60}")
    for s, m in results.items():
        print(f"  {s:5s}  R²={m['R2']:.4f}  MAE={m['MAE']:.4f}  Pearson={m['Pearson']:.4f}  Spearman={m['Spearman']:.4f}")
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
