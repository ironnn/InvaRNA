"""
2_fm_mutation.py — FM-generated UTR replacement mutations (mut201..mut(200+N))

For each WT (human only, since FM pool is human UTR), replace 5'UTR/3'UTR
with randomly sampled sequences from the FM pool. Keeps CDS unchanged.

This step is a no-op if 'human' is not in config k80.species.

Output: {output_dir}/k80/fm_mut_human.pkl
"""
import os, time, yaml, glob
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
cfg_path = os.environ.get(
    "INVARNA_GENERATION_CONFIG",
    os.environ.get("ABLATION_CONFIG", os.path.join(ROOT_DIR, "data_generation", "flow_matching", "configs", "flow_matching.yaml")),
)
with open(cfg_path) as f:
    CFG = yaml.safe_load(f)

SPECIES_LIST = CFG["species"]
N_FM = int(CFG["n_variants_per_anchor"])
N_GENES = CFG.get("n_genes", None)
RANDOM_SEED = int(CFG.get("random_seed", 42))



def resolve_path(path):
    return path if os.path.isabs(path) else os.path.join(ROOT_DIR, path)


ASSETS_DIR = resolve_path(CFG.get("assets_dir", "assets/training_data/synthetic"))
OUTPUT_DIR = os.path.join(ROOT_DIR, CFG["output_dir"], "k80")

WT_HUMAN = resolve_path(CFG.get("wt_human_path", os.path.join(ASSETS_DIR, "wt_human.pkl")))
FM_POOL_DIR = resolve_path(CFG.get("fm_pool_dir", os.path.join(ASSETS_DIR, "fm_seq")))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "fm_mut_human.pkl")


def load_fm_pool():
    """Load all FM-generated sequences from assets/fm_seq/*.csv"""
    csv_paths = sorted(glob.glob(os.path.join(FM_POOL_DIR, "**", "*.csv"), recursive=True))
    if not csv_paths:
        raise FileNotFoundError(f"No FM CSVs found in {FM_POOL_DIR}")
    dfs = []
    for p in csv_paths:
        tmp = pd.read_csv(p)
        if "val_seq_infer" in tmp.columns:
            dfs.append(tmp[["val_seq_infer"]])
    if not dfs:
        raise RuntimeError("FM CSVs do not contain 'val_seq_infer' column")
    pool = pd.concat(dfs, ignore_index=True)
    # Strip trailing N's (as in original fmseq.py)
    cleaned = pool["val_seq_infer"].str.rstrip("N")
    pool["utr5"] = cleaned.str.slice(0, 500)
    pool["utr3"] = cleaned.str.slice(-1000)
    return pool


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("Step 2: FM UTR replacement mutations")
    print("=" * 60)

    if "human" not in SPECIES_LIST:
        print("  'human' not in species list; skipping FM mutation.")
        # Save an empty placeholder so merge step is happy
        pd.DataFrame(columns=["transcript_id", "gene_id", "species", "mrna",
                              "utr5_size", "cds_size", "utr3_size",
                              "mean_te", "mut_step"]).to_pickle(OUTPUT_FILE)
        return

    t0 = time.time()

    # Load human WT
    print(f"Loading human WT: {WT_HUMAN}")
    df_wt = pd.read_pickle(WT_HUMAN).copy()
    # Normalize column names
    if "humanTE" in df_wt.columns:
        df_wt = df_wt.rename(columns={"humanTE": "mean_te"})
    if "human_gene_id" in df_wt.columns:
        df_wt = df_wt.rename(columns={"human_gene_id": "gene_id"})

    # split from mrna if needed
    if "cds" not in df_wt.columns:
        df_wt["utr5"] = df_wt.apply(lambda r: r["mrna"][: r["utr5_size"]], axis=1)
        df_wt["cds"] = df_wt.apply(lambda r: r["mrna"][r["utr5_size"]: r["utr5_size"] + r["cds_size"]], axis=1)
        df_wt["utr3"] = df_wt.apply(lambda r: r["mrna"][r["utr5_size"] + r["cds_size"]:], axis=1)

    df_wt["species"] = "human"
    if N_GENES is not None:
        df_wt = df_wt.iloc[:N_GENES]
    print(f"  WT rows: {len(df_wt):,}")

    # Load FM pool
    print("Loading FM pool...")
    pool = load_fm_pool()
    print(f"  FM pool size: {len(pool):,}")

    # Repeat each WT N times
    print(f"Expanding {N_FM}x per WT...")
    df_exp = df_wt.loc[df_wt.index.repeat(N_FM)].reset_index(drop=True)

    rng = np.random.default_rng(RANDOM_SEED)
    rand_idx = rng.choice(pool.index, len(df_exp), replace=True)
    sampled = pool.iloc[rand_idx].reset_index(drop=True)
    df_exp["utr5"] = sampled["utr5"].values
    df_exp["utr3"] = sampled["utr3"].values
    df_exp["utr5_size"] = df_exp["utr5"].str.len()
    df_exp["utr3_size"] = df_exp["utr3"].str.len()
    df_exp["mrna"] = df_exp["utr5"] + df_exp["cds"] + df_exp["utr3"]

    # mut IDs: mut201 .. mut(200+N_FM)
    suffixes = np.tile(np.arange(201, 201 + N_FM), len(df_wt))
    base_ids = df_exp["transcript_id"].astype(str).str.replace(r"_mut0$", "", regex=True)
    df_exp["transcript_id"] = base_ids + "_mut" + suffixes.astype(str)
    df_exp["mut_step"] = suffixes

    keep_cols = ["transcript_id", "gene_id", "species", "mrna",
                 "utr5_size", "cds_size", "utr3_size", "utr5", "cds", "utr3",
                 "mean_te", "mut_step"]
    keep_cols = [c for c in keep_cols if c in df_exp.columns]

    df_exp[keep_cols].to_pickle(OUTPUT_FILE)
    print(f"\n  Output rows: {len(df_exp):,}")
    print(f"  Saved: {OUTPUT_FILE}")
    print(f"  Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
