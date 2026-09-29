#!/usr/bin/env python
# encoding: utf-8
"""
RiboNN full-parameter fine-tuning on the pure_te_bench TE dataset.

Same idea as the UTR-LM / backbone fine-tunes: take the RiboNN INFERENCE weights
(one nested-CV fold, the same checkpoints used by InvaRNA/inference/infer.py ->
deps/ribonn -> RiboNN-1.0.0/models/<run_id>/state_dict.pth), UNFREEZE everything,
and fine-tune on this dataset's mean_te label.

RiboNN is a Saluki-style 1-D CNN (not a transformer):
  - input: one-hot transcript (A/T/C/G) + codon-label channels -> (C=5, L=13318),
    5'-padded & aligned at start codon (RiboNN's own DataFrameDataset does this).
  - fold0 head outputs 78 cell-line TEs; we replace the final Linear 64->78 with
    64->1 (re-init) to regress the single mean_te, keeping all pretrained conv weights.

Records follow sft/train_te_evo2.py:
  per-epoch val+test -> sklearn r2_score -> {save_dir}/{split}_ep{N}_r2_{r2}.csv,
  metrics.csv, best-by-val-R2 best.pt.

Env: mamballm.  Single process (RiboNN is small; ~6M params).

Run:
  conda run -n mamballm python sft/train_te_ribonn.py \
      --epochs 10 --bsz 16 --lr 1e-4 --output_dir ./te_ribonn_out
"""
import os, sys, csv, json, time, argparse
os.environ.setdefault("USE_TF", "0")
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

RIBONN_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", "checkpoints", "external", "pretrained", "models", "ribonn", "RiboNN-1.0.0")
sys.path.insert(0, RIBONN_ROOT)
from src.utils.helpers import extract_config
from src.data import DataFrameDataset
from src.model import RiboNN

MAX_UTR5_LEN = 1381
MAX_CDS_UTR3_LEN = 11937


def load_split(path):
    import pyarrow.parquet as pq
    import re
    t = pq.read_table(path, columns=["mrna", "utr5_size", "cds_size", "mean_te"]).to_pandas()
    # RiboNN's base_index only covers A/T/C/G; map any ambiguity base (N, etc.) -> A
    # to keep length/alignment intact (negligible: ~1 base in 1 of 20350 seqs).
    t["mrna"] = t["mrna"].str.upper().str.replace("U", "T").str.replace(r"[^ACGT]", "A", regex=True)
    t["tx_sequence"] = t["mrna"]
    # RiboNN has a fixed grid: utr5 <= MAX_UTR5_LEN, cds+utr3 <= MAX_CDS_UTR3_LEN
    # (5'-padded, aligned at the start codon). ~3% of pure_te_bench overflows it;
    # truncate those to fit — clip the 5'UTR from its 5' end, and clip cds+utr3 at the 3' end.
    def _fit(row):
        seq, u5, cds = row["mrna"].upper().replace("U", "T"), int(row["utr5_size"]), int(row["cds_size"])
        utr3 = len(seq) - u5 - cds
        u5_seq, cds_seq, u3_seq = seq[:u5], seq[u5:u5 + cds], seq[u5 + cds:]
        if u5 > MAX_UTR5_LEN:                      # keep the 3'-most (proximal) part of 5'UTR
            u5_seq = u5_seq[-MAX_UTR5_LEN:]; u5 = MAX_UTR5_LEN
        budget = MAX_CDS_UTR3_LEN
        if len(cds_seq) > budget:                  # extreme: CDS alone overflows -> clip CDS
            cds_seq = cds_seq[:budget]; u3_seq = ""
        else:
            u3_seq = u3_seq[:budget - len(cds_seq)]
        tx = u5_seq + cds_seq + u3_seq
        return pd.Series({"tx_sequence": tx, "utr5_size": len(u5_seq), "cds_size": len(cds_seq)})
    over = (t["utr5_size"] > MAX_UTR5_LEN) | ((t["cds_size"] + (t["mrna"].str.len() - t["utr5_size"] - t["cds_size"])) > MAX_CDS_UTR3_LEN)
    if over.any():
        fixed = t[over].apply(_fit, axis=1)
        t.loc[over, ["tx_sequence", "utr5_size", "cds_size"]] = fixed[["tx_sequence", "utr5_size", "cds_size"]].values
    t["tx_size"] = t["tx_sequence"].str.len()
    t = t.reset_index(drop=True)
    return t


def make_dataset(df, cfg):
    return DataFrameDataset(
        df, "mean_te", MAX_UTR5_LEN, MAX_CDS_UTR3_LEN, int(df.tx_size.max()),
        cfg["pad_5_prime"], cfg["split_utr5_cds_utr3_channels"],
        cfg["label_codons"], cfg.get("label_3rd_nt_of_codons", False),
        cfg["label_utr5"], cfg["label_utr3"], cfg["label_splice_sites"], cfg["label_up_probs"])


@torch.no_grad()
def predict(model, ds, bsz, device):
    model.eval()
    preds = []
    dl = DataLoader(ds, batch_size=bsz, shuffle=False, num_workers=4)
    for x, _ in dl:
        out = model(x.to(device))            # (N, 1)
        preds.extend(out.squeeze(-1).float().cpu().tolist())
    model.train()
    return np.asarray(preds, dtype=np.float64)


def eval_dump(model, ds, labs, bsz, device, split, epoch, save_dir):
    preds = predict(model, ds, bsz, device)
    tgt = np.asarray(labs, dtype=np.float64)
    r2 = float(r2_score(tgt, preds)); pear = float(pearsonr(preds, tgt)[0])
    spear = float(spearmanr(preds, tgt).statistic); mse = float(((preds - tgt) ** 2).mean())
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{split}_ep{epoch}_r2_{r2:.4f}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pred", "target"])
        for p, t in zip(preds, tgt): w.writerow([f"{p:.6f}", f"{t:.6f}"])
    return {"split": split, "epoch": epoch, "r2": r2, "pearson": pear, "spearman": spear, "mse": mse}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="assets/training_data/common_te_splits")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--output_dir", default="./te_ribonn_out")
    ap.add_argument("--seed", type=int, default=2222)
    ap.add_argument("--max_train", type=int, default=-1)
    args = ap.parse_args()

    if torch.cuda.is_available() and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    jlog = open(os.path.join(args.output_dir, "train_log.jsonl"), "a")
    mcsv = open(os.path.join(args.output_dir, "metrics.csv"), "a", newline="")
    mw = csv.writer(mcsv)
    if mcsv.tell() == 0:
        mw.writerow(["epoch", "split", "r2", "pearson", "spearman", "mse"]); mcsv.flush()
    def jl(d):
        d["t"] = time.strftime("%H:%M:%S"); print("  " + json.dumps(d), flush=True)
        jlog.write(json.dumps(d) + "\n"); jlog.flush()

    # --- config + fold0 weights ---
    runs = pd.read_csv(os.path.join(RIBONN_ROOT, "models", "runs.csv")).head(1)
    run_id = runs.run_id.iloc[0]
    cfg = extract_config(runs, run_id)
    cfg["max_utr5_len"] = MAX_UTR5_LEN; cfg["max_cds_utr3_len"] = MAX_CDS_UTR3_LEN
    print(f"[TE-ribonn] fold0 run_id={run_id} num_targets={cfg['num_targets']} device={device}", flush=True)

    model = RiboNN(**cfg)
    sd = torch.load(os.path.join(RIBONN_ROOT, "models", run_id, "state_dict.pth"), map_location="cpu")
    model.load_state_dict(sd)
    # swap 78-target head -> 1 (mean_te); keep pretrained conv stack
    in_f = model.head[-1].in_features
    model.head[-1] = torch.nn.Linear(in_f, 1)
    model.num_targets = 1
    model.loss = torch.nn.functional.mse_loss   # single target, no NA masking needed
    model = model.to(device)
    for p in model.parameters(): p.requires_grad = True
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[full-FT] trainable {n_tr:,} (head reinit 64->1, all conv unfrozen)", flush=True)

    # --- data ---
    print("[data] loading...", flush=True)
    tr = pd.concat([load_split(os.path.join(args.data_dir, "human_train_wt.parquet")),
                    load_split(os.path.join(args.data_dir, "mouse_train_wt.parquet"))], ignore_index=True)
    val = load_split(os.path.join(args.data_dir, "human_val_wt.parquet"))
    test = load_split(os.path.join(args.data_dir, "human_test_wt.parquet"))
    if args.max_train > 0:
        tr = tr.head(args.max_train); val = val.head(64); test = test.head(64)
    print(f"[data] train={len(tr)} val={len(val)} test={len(test)}", flush=True)
    tr_ds = make_dataset(tr, cfg); val_ds = make_dataset(val, cfg); test_ds = make_dataset(test, cfg)
    val_y = val.mean_te.to_numpy(); test_y = test.mean_te.to_numpy()

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.lr, weight_decay=args.weight_decay)
    n = len(tr_ds); steps_per_epoch = max(1, n // args.bsz)
    total_steps = args.epochs * steps_per_epoch; warmup = max(1, int(args.warmup_ratio * total_steps))
    def lr_at(s):
        if s < warmup: return args.lr * s / warmup
        prog = (s - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))

    best = {"r2": -1e9, "epoch": -1}; opt_step = 0; t0 = time.time()
    print(f"[train] {args.epochs} ep x {steps_per_epoch} steps, warmup={warmup}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        dl = DataLoader(tr_ds, batch_size=args.bsz, shuffle=True, num_workers=4, drop_last=False)
        running = 0.0; nb = 0
        for x, y in dl:
            x = x.to(device); y = y.to(device)            # y: (N,1)
            out = model(x)                                 # (N,1)
            loss = torch.nn.functional.mse_loss(out, y)
            opt.zero_grad(); loss.backward()
            for g in opt.param_groups: g["lr"] = lr_at(opt_step)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            opt.step(); opt_step += 1; running += loss.item(); nb += 1
            if opt_step % 50 == 0:
                jl({"epoch": epoch, "opt_step": opt_step, "loss": round(running / max(1, nb), 4),
                    "lr": round(lr_at(opt_step), 8), "seq_per_s": round(opt_step * args.bsz / (time.time() - t0), 1)})
                running = 0.0; nb = 0
        vm = eval_dump(model, val_ds, val_y, args.bsz, device, "val", epoch, args.output_dir)
        tm = eval_dump(model, test_ds, test_y, args.bsz, device, "test", epoch, args.output_dir)
        print(f"  val  ep{epoch}: R2={vm['r2']:.4f} Pearson={vm['pearson']:.4f} Spearman={vm['spearman']:.4f} MSE={vm['mse']:.4f}", flush=True)
        print(f"  test ep{epoch}: R2={tm['r2']:.4f} Pearson={tm['pearson']:.4f} Spearman={tm['spearman']:.4f} MSE={tm['mse']:.4f}", flush=True)
        for row in (vm, tm):
            mw.writerow([row["epoch"], row["split"], f"{row['r2']:.4f}", f"{row['pearson']:.4f}", f"{row['spearman']:.4f}", f"{row['mse']:.4f}"])
        mcsv.flush(); jl({"epoch_eval": epoch, "val": vm, "test": tm})
        if vm["r2"] > best["r2"]:
            best = {"r2": vm["r2"], "epoch": epoch, "test_r2": tm["r2"]}
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best.pt"))
            jl({"saved_best": True, "epoch": epoch, "val_r2": round(vm["r2"], 4), "test_r2": round(tm["r2"], 4)})
    print("=" * 60)
    print(f"BEST val R2={best['r2']:.4f} @ epoch {best['epoch']} (test R2={best.get('test_r2', float('nan')):.4f})")
    print("=" * 60)
    json.dump(best, open(os.path.join(args.output_dir, "best_summary.json"), "w"), indent=2)
    jlog.close(); mcsv.close()


if __name__ == "__main__":
    main()
