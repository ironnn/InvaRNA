"""
3_merge.py — Merge K80 (mut0..200) and FM (mut201..215) mutations

Combines per-species K80 mutants with human FM mutants into a single pkl
consumed by downstream embedding/features steps.

Output: {output_dir}/k80/k80_mutations.pkl
"""
import os, time, yaml, gc
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
cfg_path = os.environ.get(
    "INVARNA_GENERATION_CONFIG",
    os.environ.get("ABLATION_CONFIG", os.path.join(ROOT_DIR, "configs", "synthetic", "k80.yaml")),
)
with open(cfg_path) as f:
    CFG = yaml.safe_load(f)

SPECIES_LIST = CFG["k80"]["species"]
OUTPUT_DIR = os.path.join(ROOT_DIR, CFG["output_dir"], "k80")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "k80_mutations.pkl")

KEEP_COLS = ["transcript_id", "gene_id", "species", "mrna",
             "utr5_size", "cds_size", "utr3_size", "mean_te", "mut_step"]


def main():
    print("=" * 60)
    print("Step 3: Merge K80 + FM mutations")
    print("=" * 60)
    t0 = time.time()

    parts = []
    for sp in SPECIES_LIST:
        p = os.path.join(OUTPUT_DIR, f"k80_mut_{sp}.pkl")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing K80 output: {p}")
        print(f"  Loading {p}...")
        df = pd.read_pickle(p)
        df = df[[c for c in KEEP_COLS if c in df.columns]]
        print(f"    rows: {len(df):,}")
        parts.append(df)

    fm_path = os.path.join(OUTPUT_DIR, "fm_mut_human.pkl")
    if os.path.exists(fm_path):
        print(f"  Loading {fm_path}...")
        df_fm = pd.read_pickle(fm_path)
        if len(df_fm) > 0:
            df_fm = df_fm[[c for c in KEEP_COLS if c in df_fm.columns]]
            print(f"    rows: {len(df_fm):,}")
            parts.append(df_fm)
        else:
            print("    (empty, skipping)")

    df_all = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    # QC: start/stop codon check (sample)
    sample = df_all.sample(min(1000, len(df_all)), random_state=42)
    atg_ok = sum(r["mrna"][r["utr5_size"]:r["utr5_size"]+3] == "ATG" for _, r in sample.iterrows())
    stop_ok = sum(r["mrna"][r["utr5_size"]+r["cds_size"]-3:r["utr5_size"]+r["cds_size"]] in ("TAA","TAG","TGA")
                  for _, r in sample.iterrows())

    print(f"\n  Total rows: {len(df_all):,}")
    print(f"  Species: {df_all['species'].value_counts().to_dict()}")
    print(f"  mut_step range: [{df_all['mut_step'].min()}, {df_all['mut_step'].max()}]")
    print(f"  Start codon ATG sample: {atg_ok}/{len(sample)}")
    print(f"  Stop codon sample:     {stop_ok}/{len(sample)}")

    df_all.to_pickle(OUTPUT_FILE)
    print(f"\n  Saved: {OUTPUT_FILE}")
    print(f"  Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
