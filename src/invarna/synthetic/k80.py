"""
1_k80_mutation.py — Real K80 evolutionary mutations (mut0 + mut1..mutN)

Supports:
  - species: ["human"], ["mouse"], or ["human", "mouse"]
  - K80 transition matrix (kappa, t_max, step_size)
  - Conservation-weighted mutation (UTRs, human only — needs cons columns)
  - Synonymous codon constraint for CDS (with optional codon usage bias)
  - OR free K80 on CDS if synonymous_cds=false

Output: {output_dir}/k80/k80_mut_{species}.pkl
"""
import os, random, time, yaml, gc
import numpy as np
import pandas as pd

from Bio.Seq import Seq
from Bio.Data import CodonTable
import python_codon_tables as pct
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
cfg_path = os.environ.get(
    "INVARNA_GENERATION_CONFIG",
    os.environ.get("ABLATION_CONFIG", os.path.join(ROOT_DIR, "data_generation", "evolutionary_variants", "configs", "k80.yaml")),
)
with open(cfg_path) as f:
    CFG = yaml.safe_load(f)

K = CFG["k80"]
SPECIES_LIST = K["species"]
KAPPA = K["kappa"]
T_MAX = K["t_max"]
STEP_SIZE = K["step_size"]
USE_CONS = K["use_conservation"]
CONS_WEIGHT = K["conservation_weight"]
CDS_MODE = K.get("cds_mode", "free")
USE_CUB = K.get("use_cub", True)
CUB_WEIGHT = float(K.get("cub_weight", 0.5))
MUT_RANGE = K.get("mut_range", None)
MUT_MOD   = K.get("mut_mod", None)
N_WORKERS = CFG["n_workers"]
N_GENES   = CFG.get("n_genes", None)



def resolve_path(path):
    return path if os.path.isabs(path) else os.path.join(ROOT_DIR, path)


ASSETS_DIR = resolve_path(CFG.get("assets_dir", "assets/training_data/synthetic"))
OUTPUT_DIR = os.path.join(ROOT_DIR, CFG["output_dir"], "k80")

WT_PATHS = {
    "human": resolve_path(CFG.get("wt_human_path", os.path.join(ASSETS_DIR, "wt_human.pkl"))),
    "mouse": resolve_path(CFG.get("wt_mouse_path", os.path.join(ASSETS_DIR, "wt_mouse.pkl"))),
}
CODON_TABLE_NAME = {"human": "h_sapiens_9606", "mouse": "m_musculus_10090"}



# =========================================================
# Precompute synonymous codon table + codon transition type counts
# =========================================================
_TRANSITIONS = {('A','G'), ('G','A'), ('C','T'), ('T','C')}
_BASES = "ACGT"
_ALL_CODONS = [a+b+c for a in _BASES for b in _BASES for c in _BASES]

def _base_type(a, b):
    """Return 'same', 'ts' (transition), or 'tv' (transversion)."""
    if a == b: return 'same'
    if (a, b) in _TRANSITIONS: return 'ts'
    return 'tv'

def precompute_synonymous_codons():
    table = CodonTable.unambiguous_dna_by_id[1]
    cache = {}
    for codon in _ALL_CODONS:
        try:
            aa = Seq(codon).translate(table=table)
            syns = [c for c in _ALL_CODONS if Seq(c).translate(table=table) == aa]
            cache[codon] = syns
        except Exception:
            cache[codon] = [codon]
    return cache

def precompute_codon_type_scores():
    """For each (src_codon, tgt_codon) pair, precompute (n_same, n_ts, n_tv).
    At runtime: score = p_same^n_same * p_ts^n_ts * p_tv^n_tv (product of per-base probs).
    Stored as dict[src][tgt] = (n_same, n_ts, n_tv).
    """
    cache = {}
    for src in _ALL_CODONS:
        row = {}
        for tgt in _ALL_CODONS:
            counts = {'same': 0, 'ts': 0, 'tv': 0}
            for j in range(3):
                counts[_base_type(src[j], tgt[j])] += 1
            row[tgt] = (counts['same'], counts['ts'], counts['tv'])
        cache[src] = row
    return cache

PRECOMPUTED_SYN = precompute_synonymous_codons()
PRECOMPUTED_TYPE_SCORES = precompute_codon_type_scores()


def codon_k80_scores(src, targets, p_same, p_ts, p_tv):
    """Vectorised K80 score for src→each target using precomputed type counts."""
    scores = np.empty(len(targets))
    type_row = PRECOMPUTED_TYPE_SCORES[src]
    for k, tgt in enumerate(targets):
        ns, nts, ntv = type_row[tgt]
        scores[k] = (p_same ** ns) * (p_ts ** nts) * (p_tv ** ntv)
    return scores


# =========================================================
# K80 matrix
# =========================================================
def compute_k80_matrix(kappa, t):
    if kappa == -1: kappa = 0
    alpha = (kappa / (kappa + 1)) * t
    beta = (1 / (kappa + 1)) * t
    exp_beta = np.exp(-4 * beta)
    exp_alpha_beta = np.exp(-2 * (alpha + beta))
    p_same = 0.25 + 0.25 * exp_beta + 0.5 * exp_alpha_beta
    p_ts = 0.25 + 0.25 * exp_beta - 0.25 * exp_alpha_beta
    p_tv = 0.25 - 0.25 * exp_beta
    mat = np.array([
        [p_same, p_tv,   p_ts, p_tv  ],
        [p_tv,   p_same, p_tv, p_ts  ],
        [p_ts,   p_tv,   p_same, p_tv],
        [p_tv,   p_ts,   p_tv, p_same],
    ])
    mat = mat / mat.sum(axis=1, keepdims=True)
    bases = "ACGT"
    return {bases[i]: {bases[j]: mat[i, j] for j in range(4)} for i in range(4)}

def compute_k80_scalars(kappa, t):
    """Return (p_same, p_ts, p_tv) for use with codon_k80_scores."""
    if kappa == -1: kappa = 0
    alpha = (kappa / (kappa + 1)) * t
    beta = (1 / (kappa + 1)) * t
    exp_beta = np.exp(-4 * beta)
    exp_alpha_beta = np.exp(-2 * (alpha + beta))
    p_same = 0.25 + 0.25 * exp_beta + 0.5 * exp_alpha_beta
    p_ts   = 0.25 + 0.25 * exp_beta - 0.25 * exp_alpha_beta
    p_tv   = 0.25 - 0.25 * exp_beta
    # normalise so per-base probs sum to 1
    total = p_same + p_ts + 2 * p_tv
    return p_same / total, p_ts / total, p_tv / total


# =========================================================
# UTR mutation (K80 + optional conservation)
# =========================================================
def mutate_utr(seq, prob_matrix, cons_scores, use_conservation, cons_weight):
    mutated = list(seq)
    n = len(seq)
    if n == 0:
        return seq

    rand_vals = np.random.random(n)
    if use_conservation and cons_scores is not None:
        scores = np.asarray(cons_scores, dtype=float)[:n]
        if len(scores) < n:
            scores = np.pad(scores, (0, n - len(scores)), "constant")
        mut_probs = np.clip(1.0 - cons_weight * scores, 0.0, 1.0)
    else:
        mut_probs = np.ones(n)

    for i, base in enumerate(seq):
        if base not in "ACGT":
            continue
        if rand_vals[i] < mut_probs[i]:
            row = prob_matrix[base]
            candidates = list(row.keys())
            probs = list(row.values())
            mutated[i] = np.random.choice(candidates, p=probs)
    return "".join(mutated)


# =========================================================
# CDS mutation (synonymous-only with CUB, or free K80)
# =========================================================
def mutate_cds_synonymous(cds_seq, k80_scalars, cons_scores, use_conservation, cons_weight, codon_bias):
    """Strict synonymous: every codon mutates to a synonymous codon via K80+CUB probs. No escape."""
    p_same, p_ts, p_tv = k80_scalars
    mutated = list(cds_seq)
    n = len(cds_seq)
    cons_arr = None
    if use_conservation and cons_scores is not None:
        cons_arr = np.asarray(cons_scores, dtype=float)
        if len(cons_arr) < n:
            cons_arr = np.pad(cons_arr, (0, n - len(cons_arr)), "constant")

    for i in range(3, n - 3, 3):
        codon = "".join(mutated[i:i+3])
        if len(codon) < 3:
            break
        syns = PRECOMPUTED_SYN.get(codon, [codon])
        if len(syns) <= 1:
            continue
        norm = codon_k80_scores(codon, syns, p_same, p_ts, p_tv)
        total = norm.sum()
        if total == 0:
            continue
        norm /= total
        if codon_bias:
            bias = np.array([codon_bias.get(c, 0.0) for c in syns])
            w = norm * bias
            if w.sum() > 0:
                norm = CUB_WEIGHT * (w / w.sum()) + (1.0 - CUB_WEIGHT) * norm

        if cons_arr is not None:
            codon_cons = float(np.mean(cons_arr[i:i+3]))
        else:
            codon_cons = 0.0
        mut_prob = max(0.0, 1.0 - cons_weight * codon_cons)

        if random.random() < mut_prob:
            new_codon = np.random.choice(syns, p=norm)
            mutated[i:i+3] = list(new_codon)
    return "".join(mutated)


def mutate_cds_free(cds_seq, prob_matrix, cons_scores, use_conservation, cons_weight):
    """Free K80 on CDS (may break protein); protects start/stop codons."""
    mutated = list(cds_seq)
    n = len(cds_seq)
    if n == 0:
        return cds_seq
    rand_vals = np.random.random(n)
    if use_conservation and cons_scores is not None:
        scores = np.asarray(cons_scores, dtype=float)[:n]
        if len(scores) < n:
            scores = np.pad(scores, (0, n - len(scores)), "constant")
        mut_probs = np.clip(1.0 - cons_weight * scores, 0.0, 1.0)
    else:
        mut_probs = np.ones(n)

    for i, base in enumerate(cds_seq):
        if i < 3 or i >= n - 3:
            continue
        if base not in "ACGT":
            continue
        if rand_vals[i] < mut_probs[i]:
            row = prob_matrix[base]
            mutated[i] = np.random.choice(list(row.keys()), p=list(row.values()))
    return "".join(mutated)


# =========================================================
# Worker: process one gene (generate mut0 + mut1..mutN)
# =========================================================
def process_one_gene(args):
    row_dict, seed, t_values, codon_bias, has_cons = args
    np.random.seed(seed)
    random.seed(seed)

    tid = row_dict["transcript_id"]
    u5_seq = row_dict["utr5"] if row_dict.get("utr5") else ""
    cds_seq = row_dict["cds"] if row_dict.get("cds") else ""
    u3_seq = row_dict["utr3"] if row_dict.get("utr3") else ""
    u5_cons = row_dict.get("utr5_cons") if has_cons else None
    cds_cons = row_dict.get("cds_cons") if has_cons else None
    u3_cons = row_dict.get("utr3_cons") if has_cons else None

    base_id = tid
    results = []

    def make_row(mut_id, u5, cds, u3):
        new_mrna = u5 + cds + u3
        return {
            "transcript_id": f"{base_id}_mut{mut_id}",
            "gene_id": row_dict["gene_id"],
            "species": row_dict["species"],
            "mean_te": row_dict["mean_te"],
            "utr5_size": len(u5),
            "cds_size": len(cds),
            "utr3_size": len(u3),
            "utr5": u5,
            "cds": cds,
            "utr3": u3,
            "mrna": new_mrna,
            "mut_step": mut_id,
        }

    # mut0: original WT
    results.append(make_row(0, u5_seq, cds_seq, u3_seq))

    # mut1..mutN
    for i, t in enumerate(t_values):
        mat = compute_k80_matrix(KAPPA, t)
        scalars = compute_k80_scalars(KAPPA, t)
        m_u5 = mutate_utr(u5_seq, mat, u5_cons, USE_CONS, CONS_WEIGHT)
        if CDS_MODE == "strict":
            m_cds = mutate_cds_synonymous(cds_seq, scalars, cds_cons, USE_CONS, CONS_WEIGHT,
                                           codon_bias if USE_CUB else None)
        else:
            m_cds = mutate_cds_free(cds_seq, mat, cds_cons, USE_CONS, CONS_WEIGHT)
        m_u3 = mutate_utr(u3_seq, mat, u3_cons, USE_CONS, CONS_WEIGHT)
        results.append(make_row(i + 1, m_u5, m_cds, m_u3))

    return results


# =========================================================
# Main: process each species
# =========================================================
def process_species(species):
    print("\n" + "=" * 60)
    print(f"Processing species: {species}")
    print("=" * 60)

    wt_path = WT_PATHS[species]
    if not os.path.exists(wt_path):
        raise FileNotFoundError(f"WT data not found: {wt_path}")

    print(f"Loading WT: {wt_path}")
    df = pd.read_pickle(wt_path)
    if N_GENES is not None:
        df = df.iloc[:N_GENES]
    print(f"  Rows: {len(df):,}, cols: {list(df.columns)}")

    # Normalize column names (human uses humanTE/human_gene_id, mouse uses mean_te/gene_id)
    rename_map = {"humanTE": "mean_te", "human_gene_id": "gene_id",
                  "mouseTE": "mean_te", "mouse_gene_id": "gene_id"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "species" not in df.columns:
        df["species"] = species

    # If utr5/cds/utr3 not present (mouse case), split from mrna
    if "utr5" not in df.columns or "cds" not in df.columns or "utr3" not in df.columns:
        print("  Splitting mrna into utr5/cds/utr3...")
        df["utr5"] = df.apply(lambda r: r["mrna"][: r["utr5_size"]], axis=1)
        df["cds"] = df.apply(lambda r: r["mrna"][r["utr5_size"]: r["utr5_size"] + r["cds_size"]], axis=1)
        df["utr3"] = df.apply(lambda r: r["mrna"][r["utr5_size"] + r["cds_size"]:], axis=1)

    has_cons = all(c in df.columns for c in ("utr5_cons", "cds_cons", "utr3_cons"))
    print(f"  Conservation columns present: {has_cons}")
    if USE_CONS and not has_cons:
        print(f"  WARNING: use_conservation=true but no cons columns for {species}; will use 0 weights")

    # Codon bias
    codon_bias = None
    if USE_CUB:
        try:
            codon_bias = pct.get_codons_table(CODON_TABLE_NAME[species])
        except Exception as e:
            print(f"  WARN: cannot load codon table: {e}")
            codon_bias = None

    # t values: mut_id = round(t / step_size), so t_values[i] → mut_id i+1
    t_values = np.arange(STEP_SIZE, T_MAX + STEP_SIZE/1000, STEP_SIZE)
    mut_ids = np.arange(1, len(t_values) + 1)

    if MUT_RANGE is not None:
        lo, hi = [int(x) for x in str(MUT_RANGE).split('-')]
        mask = (mut_ids >= lo) & (mut_ids <= hi)
        t_values = t_values[mask]
        mut_ids = mut_ids[mask]

    if MUT_MOD is not None:
        s = str(MUT_MOD)
        n, r = (int(s.split(':')[0]), int(s.split(':')[1])) if ':' in s else (int(s), 0)
        mask = (mut_ids % n == r)
        t_values = t_values[mask]
        mut_ids = mut_ids[mask]

    print(f"  N K80 mutants per gene: {len(t_values)} (mut_id {mut_ids[0]}..{mut_ids[-1]}, mut_range={MUT_RANGE}, mut_mod={MUT_MOD})")

    # Build args
    args_list = []
    for idx, (_, row) in enumerate(df.iterrows()):
        args_list.append((row.to_dict(), 42 + idx, t_values, codon_bias, has_cons))

    # Run
    print(f"\nRunning K80 mutation with {N_WORKERS} workers...")
    all_results = []
    with Pool(N_WORKERS) as pool:
        for batch in tqdm(pool.imap(process_one_gene, args_list, chunksize=5),
                          total=len(args_list), desc=f"  {species}"):
            all_results.extend(batch)

    df_out = pd.DataFrame(all_results)
    print(f"  Output rows: {len(df_out):,}")

    out_path = os.path.join(OUTPUT_DIR, f"k80_mut_{species}.pkl")
    df_out.to_pickle(out_path)
    print(f"  Saved: {out_path}")
    del df, args_list, all_results, df_out
    gc.collect()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("Step 1: K80 Evolutionary Mutation")
    print("=" * 60)
    print(f"  species: {SPECIES_LIST}")
    print(f"  kappa: {KAPPA}, t_max: {T_MAX}, step_size: {STEP_SIZE}")
    print(f"  use_conservation: {USE_CONS}, conservation_weight: {CONS_WEIGHT}")
    print(f"  cds_mode: {CDS_MODE}")

    t0 = time.time()
    for sp in SPECIES_LIST:
        if sp not in WT_PATHS:
            raise ValueError(f"Unknown species: {sp}")
        process_species(sp)
    print(f"\nAll K80 mutations done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
