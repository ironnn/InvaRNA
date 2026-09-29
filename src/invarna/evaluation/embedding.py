#!/usr/bin/env python
# encoding: utf-8
"""
embed.py — InvaRNA backbone RAW per-token embedding extraction (no post-processing).

Self-contained: loads the InvaRNA backbone from this package and runs a
single forward pass over the CDS-anchored, padded transcript. Unlike the benchmark
extractors, it does NO pooling / region-concat — it returns the raw per-token hidden
states (B, L, 512).

Alignment used by the recovered embedding workflow: start codon pinned at
FIXED_CDS_START=1000, pad/truncate to TOTAL_LENGTH=10000. Unlike the selected
Stage-2 trainer, this embedding implementation left-truncates overlength 5' UTRs;
the discrepancy is retained as a documented implementation limitation.
N -> pad token.

Output granularity (--include_pad toggle):
  default        : real span only, (L_real, 512)  -- N-pad flanks dropped, not pooled
  --include_pad  : whole aligned sequence, (10000, 512)

Input:  .pkl / .pickle / .parquet with columns: transcript_id, mrna, utr5_size, cds_size, utr3_size
Output: .pkl  ->  {"keys": [...], "embeddings": [np.ndarray(L, 512) float32, ...],
                   "include_pad": bool, "dim": 512}

Multi-GPU: --gpus 0,1,2,3 spawns one worker per GPU writing shard_{rank}.pkl, then merges.
           A single GPU (n=1) runs in-process (no spawn).

Examples:
  # single GPU
  python embed.py --input wt.parquet --output emb/raw.pkl --gpus 0 --batch_size 16
  # multi-GPU, whole padded sequence
  python embed.py --input wt.pkl --output emb/raw.pkl --gpus 0,1,2,3 --include_pad
"""
import os
import sys
import argparse
import pickle

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))

# ── compat shim so legacy caduceus.* pickled weights load, then repo backbone imports
sys.path.insert(0, os.path.join(_ROOT, "src"))
from invarna import models as backbone
from invarna.models import configuration as backbone_configuration
from invarna.models import mamba_backbone as backbone_modeling
from invarna.models import tokenization as backbone_tokenization
import types as _types
_pkg = _types.ModuleType("caduceus")
sys.modules.setdefault("caduceus", _pkg)
sys.modules.setdefault("caduceus.configuration_caduceus", backbone_configuration)
sys.modules.setdefault("caduceus.modeling_caduceus", backbone_modeling)
sys.modules.setdefault("caduceus.tokenization_caduceus", backbone_tokenization)

from invarna.models.configuration import InvaRNAConfig
from invarna.models.mamba_backbone import InvaRNAForMaskedLM
from invarna.models.tokenization import InvaRNATokenizer

TOTAL_LENGTH = 10000
FIXED_CDS_START = 1000
DEFAULT_CONFIG = os.path.join(_ROOT, "backbone", "configs", "mamba_motif_moe.yaml")
DEFAULT_WEIGHTS = os.path.join(_ROOT, "assets", "checkpoints", "backbone", "pretrained_step13500.pt")


def force_align_sequence(raw_seq, u5_len, cds_len, u3_len):
    """CDS@1000 alignment. Returns padded_seq + the real-span [start,end) (pad flanks excluded)
    + per-region [start,end) ranges (5'UTR / CDS / 3'UTR).
    Verbatim alignment from 4_extract_embedding.py."""
    seq = str(raw_seq)
    u5_truncate_offset = max(0, u5_len - FIXED_CDS_START)
    if u5_truncate_offset > 0:
        seq = seq[u5_truncate_offset:]
        current_u5_len = FIXED_CDS_START
    else:
        current_u5_len = u5_len
    left_padding_count = FIXED_CDS_START - current_u5_len
    padded_seq = ("N" * left_padding_count) + seq
    padded_seq = padded_seq[:TOTAL_LENGTH].ljust(TOTAL_LENGTH, "N")
    real_start = left_padding_count
    real_end = min(FIXED_CDS_START + cds_len + u3_len, TOTAL_LENGTH)
    range_u5 = (left_padding_count, FIXED_CDS_START)
    range_cds = (FIXED_CDS_START, min(FIXED_CDS_START + cds_len, TOTAL_LENGTH))
    range_u3 = (min(FIXED_CDS_START + cds_len, TOTAL_LENGTH),
                min(FIXED_CDS_START + cds_len + u3_len, TOTAL_LENGTH))
    return padded_seq, real_start, real_end, range_u5, range_cds, range_u3


class AlignmentDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.unk_id = tokenizer.convert_tokens_to_ids("N")
        self.pad_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        padded, rstart, rend, ru5, rcds, ru3 = force_align_sequence(
            row["mrna"], int(row["utr5_size"]), int(row["cds_size"]), int(row["utr3_size"]))
        input_ids = self.tokenizer.encode(padded, add_special_tokens=False)
        input_ids = [self.pad_id if t == self.unk_id else t for t in input_ids]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "gene_id": str(row["transcript_id"]),
            "real_start": rstart, "real_end": rend,
            "u5_rng": ru5, "cds_rng": rcds, "u3_rng": ru3,
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "gene_id": [b["gene_id"] for b in batch],
        "real_start": [b["real_start"] for b in batch],
        "real_end": [b["real_end"] for b in batch],
        "u5_rng": [b["u5_rng"] for b in batch],
        "cds_rng": [b["cds_rng"] for b in batch],
        "u3_rng": [b["u3_rng"] for b in batch],
    }


def load_table(path):
    if path.endswith((".pkl", ".pickle")):
        return pd.read_pickle(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path}")


def build_backbone(config_path, weights_path, device):
    from omegaconf import OmegaConf
    mc = OmegaConf.to_container(OmegaConf.load(config_path).model.config, resolve=True)
    mc.pop("_target_", None)
    model = InvaRNAForMaskedLM(InvaRNAConfig(**mc))
    sd = torch.load(weights_path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss or unexp:
        print(f"[embed] load_state_dict: missing={len(miss)} unexpected={len(unexp)}", flush=True)
    backbone_ = model.invaRNA
    backbone_.config.return_dict = True
    backbone_.eval()
    for p in backbone_.parameters():
        p.requires_grad_(False)
    return backbone_.to(device)


def _meanmax(h, s, e):
    """mean+max pool over h[s:e] -> (2*D,); zeros if empty span."""
    if s >= e:
        return torch.zeros(h.size(-1) * 2, device=h.device)
    region = h[s:e]
    return torch.cat([region.mean(dim=0), region.max(dim=0)[0]])


def compute_multi_pools(h, u5_rng, cds_rng, u3_rng, real_start, real_end):
    """One per-token hidden state h (L,D) -> dict of named fixed-length vectors (all mean+max)."""
    L = h.size(0)
    u5_s, u5_e = u5_rng
    return {
        # 5'UTR/CDS/3'UTR region mean+max, 5'UTR excludes left N-pad
        "region_nopad": torch.cat([_meanmax(h, u5_s, u5_e),
                                   _meanmax(h, *cds_rng),
                                   _meanmax(h, *u3_rng)]),
        # same but 5'UTR includes the left N-pad flank [0, 1000)
        "region_withpad": torch.cat([_meanmax(h, 0, u5_e),
                                     _meanmax(h, *cds_rng),
                                     _meanmax(h, *u3_rng)]),
        # whole real span (no region split), pad excluded
        "span_nopad": _meanmax(h, real_start, real_end),
        # whole aligned sequence incl all N-pad
        "span_withpad": _meanmax(h, 0, L),
    }


def run_worker(rank, gpu_id, df, args, shard_path):
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{gpu_id}")
    bb = build_backbone(args.config, args.init_weights, device)
    tokenizer = InvaRNATokenizer(model_max_length=TOTAL_LENGTH)
    ds = AlignmentDataset(df, tokenizer)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)

    keys, embs = [], []
    pools = {}  # name -> list of np arrays (multi_pool mode)
    try:
        from tqdm import tqdm
        it = tqdm(loader, desc=f"rank{rank}", position=rank) if rank == 0 else loader
    except Exception:
        it = loader

    for batch in it:
        input_ids = batch["input_ids"].to(device)
        with torch.no_grad():
            hidden = bb(input_ids=input_ids)[0]  # (B, L, 512) raw per-token hidden states
        for i, gid in enumerate(batch["gene_id"]):
            h = hidden[i]
            if args.multi_pool:
                pd_ = compute_multi_pools(
                    h, batch["u5_rng"][i], batch["cds_rng"][i], batch["u3_rng"][i],
                    batch["real_start"][i], batch["real_end"][i])
                keys.append(gid)
                for name, vec in pd_.items():
                    pools.setdefault(name, []).append(vec.float().cpu().numpy())
            elif args.region_pool:
                # 5'UTR / CDS / 3'UTR region mean+max pooling -> fixed 6*512=3072 vector.
                # region_withpad: 5'UTR span starts at 0 (includes left N-pad flank).
                u5_s, u5_e = batch["u5_rng"][i]
                if args.region_withpad:
                    u5_s = 0
                emb = torch.cat([_meanmax(h, u5_s, u5_e),
                                 _meanmax(h, *batch["cds_rng"][i]),
                                 _meanmax(h, *batch["u3_rng"][i])])
                keys.append(gid)
                embs.append(emb.float().cpu().numpy())
            else:
                if not args.include_pad:
                    s, e = batch["real_start"][i], batch["real_end"][i]
                    h = h[s:e]
                keys.append(gid)
                embs.append(h.float().cpu().numpy())

    with open(shard_path, "wb") as f:
        if args.multi_pool:
            pickle.dump({"keys": keys, "pools": {k: np.stack(v) for k, v in pools.items()}}, f)
        else:
            pickle.dump({"keys": keys, "embeddings": embs}, f)
    print(f"[embed] rank{rank} wrote {len(keys)} embeddings -> {shard_path}", flush=True)


def _mp_entry(rank, gpu_ids, df_splits, args, shard_dir):
    run_worker(rank, gpu_ids[rank], df_splits[rank], args, os.path.join(shard_dir, f"shard_{rank}.pkl"))


def main():
    ap = argparse.ArgumentParser(description="InvaRNA raw per-token embedding extraction")
    ap.add_argument("--input", required=True, help=".pkl/.pickle/.parquet with transcript_id,mrna,utr5_size,cds_size,utr3_size")
    ap.add_argument("--output", required=True, help="output .pkl")
    ap.add_argument("--gpus", default="0", help="comma-separated GPU ids, e.g. 0,1,2,3")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--include_pad", action="store_true",
                    help="return whole aligned (10000,512) incl N-pad; default = real span only")
    ap.add_argument("--region_pool", action="store_true",
                    help="5'UTR/CDS/3'UTR region mean+max pooling -> fixed (3072,) vector per seq "
                         "(for LightGBM teacher); overrides --include_pad")
    ap.add_argument("--multi_pool", action="store_true",
                    help="one forward pass -> dict of named fixed-length pools "
                         "(region_nopad/region_withpad/span_nopad/span_withpad); for ablation")
    ap.add_argument("--region_withpad", action="store_true",
                    help="with --region_pool: 5'UTR span includes the left N-pad flank "
                         "(start=0 instead of real 5'UTR start)")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--init_weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--limit", type=int, default=-1, help="debug: only first N rows")
    args = ap.parse_args()

    df = load_table(args.input)
    need = {"transcript_id", "mrna", "utr5_size", "cds_size", "utr3_size"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"input missing columns: {missing}")
    if args.limit > 0:
        df = df.head(args.limit)
    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    print(f"[embed] rows={len(df)} gpus={gpu_ids} include_pad={args.include_pad}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    shard_dir = args.output + "_shards"
    os.makedirs(shard_dir, exist_ok=True)

    if len(gpu_ids) == 1:
        # single GPU: run in-process, no spawn
        run_worker(0, gpu_ids[0], df, args, os.path.join(shard_dir, "shard_0.pkl"))
    else:
        splits = np.array_split(df, len(gpu_ids))
        import torch.multiprocessing as mp
        mp.spawn(_mp_entry, args=(gpu_ids, splits, args, shard_dir), nprocs=len(gpu_ids), join=True)

    # merge shards in rank order
    if args.multi_pool:
        keys, pools = [], {}
        for rank in range(len(gpu_ids)):
            with open(os.path.join(shard_dir, f"shard_{rank}.pkl"), "rb") as f:
                d = pickle.load(f)
            keys.extend(d["keys"])
            for name, arr in d["pools"].items():
                pools.setdefault(name, []).append(arr)
        pools = {name: np.concatenate(chunks, axis=0) for name, chunks in pools.items()}
        out = {"keys": keys, "pools": pools,
               "dims": {name: arr.shape[1] for name, arr in pools.items()}}
        with open(args.output, "wb") as f:
            pickle.dump(out, f)
        print(f"[embed] DONE: {len(keys)} seqs -> {args.output} "
              f"pools={ {k: v.shape for k, v in pools.items()} }", flush=True)
        return

    keys, embs = [], []
    for rank in range(len(gpu_ids)):
        with open(os.path.join(shard_dir, f"shard_{rank}.pkl"), "rb") as f:
            d = pickle.load(f)
        keys.extend(d["keys"]); embs.extend(d["embeddings"])

    out = {"keys": keys, "embeddings": embs, "include_pad": args.include_pad, "dim": 512}
    with open(args.output, "wb") as f:
        pickle.dump(out, f)
    print(f"[embed] DONE: {len(keys)} embeddings -> {args.output} "
          f"(shape e.g. {embs[0].shape if embs else 'NA'})", flush=True)


if __name__ == "__main__":
    main()
