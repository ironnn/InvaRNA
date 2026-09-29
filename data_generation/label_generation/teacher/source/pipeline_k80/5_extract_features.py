"""
3_extract_features.py — Handcrafted features (CPU multiprocess)
"""
import os, sys, gc, time, yaml
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
cfg_path = os.environ.get("ABLATION_CONFIG", os.path.join(ROOT_DIR, "config.yaml"))
with open(cfg_path) as f:
    CFG = yaml.safe_load(f)

sys.path.insert(0, os.path.join(ROOT_DIR, "utils"))
from lgbm_feature_extract_from_str import fe, get_cols

OUTPUT_DIR = os.path.join(ROOT_DIR, CFG["output_dir"], "k80")
INPUT_FILE = os.path.join(OUTPUT_DIR, "k80_mutations.pkl")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "features.parquet")
NUM_CORES = CFG["n_workers"]

FEATURE_LIST = ['LL', 'P5', 'P3', 'CF', 'AAF', '3mer_freq_5', 'K', 'DC', 'Struct']


def process_feature_chunk(df_chunk):
    results = []
    mrnas = df_chunk['mrna'].values
    utr5s = df_chunk['utr5_size'].values
    cdss = df_chunk['cds_size'].values
    utr3s = df_chunk['utr3_size'].values
    lens = [len(m) for m in mrnas]
    for m, u5, c, u3, l in zip(mrnas, utr5s, cdss, utr3s, lens):
        results.append(fe(FEATURE_LIST, m, u5, c, u3, l))
    return results


def main():
    t0 = time.time()
    print("=" * 60)
    print(f"Step 3: Extract Handcrafted Features (cores={NUM_CORES})")
    print("=" * 60)

    print(f"\nLoading {INPUT_FILE}...")
    df = pd.read_pickle(INPUT_FILE)
    meta_cols = ['transcript_id', 'gene_id', 'species', 'mean_te',
                 'utr5_size', 'cds_size', 'utr3_size', 'mrna', 'mut_step']
    df_meta = df[meta_cols].copy()
    del df; gc.collect()
    print(f"  Rows: {len(df_meta):,}")

    df_chunks = [df_meta.iloc[idx] for idx in np.array_split(range(len(df_meta)), NUM_CORES * 4)]
    with Pool(NUM_CORES) as pool:
        chunk_results = list(tqdm(pool.imap(process_feature_chunk, df_chunks),
                                  total=len(df_chunks), desc="  Features"))

    all_features = []
    for res in chunk_results:
        all_features.extend(res)
    del chunk_results; gc.collect()

    col_names = get_cols(FEATURE_LIST)
    df_features = pd.DataFrame(all_features, columns=col_names)
    del all_features; gc.collect()

    df_final = pd.concat([df_meta.reset_index(drop=True), df_features], axis=1)
    del df_meta, df_features; gc.collect()

    df_final.to_parquet(OUTPUT_FILE, index=False)
    print(f"  Rows: {len(df_final):,}, Cols: {len(df_final.columns)}")
    print(f"  Saved: {OUTPUT_FILE}")
    print(f"  Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
