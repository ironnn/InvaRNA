#!/usr/bin/env python
# encoding: utf-8
"""
evo2 (StripedHyena 7B) full-parameter fine-tuning on the pure_te_bench TE dataset.

Convention follows InvaRNA/train/finetune_stage2.py:
  - label = mean_te (regression), loss = MSELoss
  - EVERY epoch: run inference on BOTH val and test, compute R2 with
    sklearn.metrics.r2_score, and dump predictions to
        {save_dir}/{split}_ep{epoch}_r2_{r2:.4f}.csv   (columns: pred,target)
  - per-epoch R2 recorded to metrics.csv / train_log.jsonl

Train : human_train_wt + mouse_train_wt   (20350)
Val   : human_val_wt  (1115)   Test : human_test_wt (1115)
Input : mrna truncated to first --max_len nt (5'); Target: mean_te.

Backbone is pipeline-sharded across all visible GPUs by vortex (single process —
NOT torchrun). bsz=1 (vortex runs one GPU at a time; batching gives no speedup).

Run:
  conda run -n mamballm python sft/train_te_evo2.py \
      --max_len 8000 --epochs 10 --lr 1e-5 --grad_accum 16 --output_dir ./te_evo2_out
"""
import os
import sys
import csv
import json
import time
import argparse
os.environ.setdefault("USE_TF", "0")
import numpy as np
import torch
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

LLM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(LLM_ROOT))
from external_models.common.evo2_ft import Evo2ForSeqCls


def load_split(path, max_len):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["mrna", "mean_te"]).to_pandas()
    seqs = [s[:max_len] for s in t["mrna"].tolist()]
    labs = t["mean_te"].astype("float32").to_numpy()
    return seqs, labs


@torch.no_grad()
def predict(model, seqs, max_len):
    """Return np.array of predictions (one scalar per sequence)."""
    model.eval()
    preds = []
    for s in seqs:
        ids, attn = model.encode([s], max_len=max_len)
        logits, _ = model(ids, attention_mask=attn, labels=None)
        preds.append(float(logits.squeeze(-1).float().cpu().item()))
    model.train()
    return np.asarray(preds, dtype=np.float64)


def eval_and_dump(model, seqs, labs, max_len, split, epoch, save_dir):
    """Predict on a split, compute sklearn R2 (+pearson/spearman), dump CSV. Returns metrics dict."""
    preds = predict(model, seqs, max_len)
    targets = np.asarray(labs, dtype=np.float64)
    r2 = float(r2_score(targets, preds))
    pear = float(pearsonr(preds, targets)[0])
    spear = float(spearmanr(preds, targets).statistic)
    mse = float(((preds - targets) ** 2).mean())

    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f"{split}_ep{epoch}_r2_{r2:.4f}.csv")
    with open(fname, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pred", "target"])
        for p, t in zip(preds, targets):
            w.writerow([f"{p:.6f}", f"{t:.6f}"])
    return {"split": split, "epoch": epoch, "r2": r2, "pearson": pear,
            "spearman": spear, "mse": mse, "csv": os.path.basename(fname)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="assets/training_data/common_te_splits")
    ap.add_argument("--max_len", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--output_dir", default="./te_evo2_out")
    ap.add_argument("--seed", type=int, default=2222)
    ap.add_argument("--max_train", type=int, default=-1, help="cap train size for quick sanity")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    jlog = open(os.path.join(args.output_dir, "train_log.jsonl"), "a")
    mcsv_path = os.path.join(args.output_dir, "metrics.csv")
    new_mcsv = not os.path.exists(mcsv_path)
    mcsv = open(mcsv_path, "a", newline="")
    mwriter = csv.writer(mcsv)
    if new_mcsv:
        mwriter.writerow(["epoch", "split", "r2", "pearson", "spearman", "mse"]); mcsv.flush()

    def jl(d):
        d["t"] = time.strftime("%H:%M:%S")
        print("  " + json.dumps(d), flush=True)
        jlog.write(json.dumps(d) + "\n"); jlog.flush()

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    # ---- data ----
    print("[data] loading parquets...", flush=True)
    h_s, h_y = load_split(os.path.join(args.data_dir, "human_train_wt.parquet"), args.max_len)
    m_s, m_y = load_split(os.path.join(args.data_dir, "mouse_train_wt.parquet"), args.max_len)
    tr_s = h_s + m_s
    tr_y = np.concatenate([h_y, m_y])
    val_s, val_y = load_split(os.path.join(args.data_dir, "human_val_wt.parquet"), args.max_len)
    test_s, test_y = load_split(os.path.join(args.data_dir, "human_test_wt.parquet"), args.max_len)
    if args.max_train > 0:
        tr_s, tr_y = tr_s[:args.max_train], tr_y[:args.max_train]
        val_s, val_y = val_s[:64], val_y[:64]
        test_s, test_y = test_s[:64], test_y[:64]
    n = len(tr_s)
    print(f"[data] train={n} (human {len(h_s)} + mouse {len(m_s)}), "
          f"val={len(val_s)}, test={len(test_s)} | max_len={args.max_len}", flush=True)

    # ---- model ----
    print("[model] loading evo2_7b (vortex sharded)...", flush=True)
    model = Evo2ForSeqCls(num_labels=1, task_type="regression", pooling="mean")
    model.train()
    n_train, n_total = model.trainable_param_report()
    print(f"[full-FT] trainable {n_train:,}/{n_total:,} ({100*n_train/n_total:.1f}%) on "
          f"{torch.cuda.device_count()} GPUs", flush=True)

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.lr, weight_decay=args.weight_decay)
    opt_steps_per_epoch = max(1, n // args.grad_accum)
    total_opt_steps = args.epochs * opt_steps_per_epoch
    warmup = max(1, int(args.warmup_ratio * total_opt_steps))
    def lr_at(s):
        if s < warmup:
            return args.lr * s / warmup
        prog = (s - warmup) / max(1, total_opt_steps - warmup)
        return args.lr * 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))

    print(f"[train] {args.epochs} epochs x {opt_steps_per_epoch} opt-steps "
          f"(grad_accum={args.grad_accum}), warmup={warmup}", flush=True)

    best = {"r2": -1e9, "epoch": -1}
    opt_step = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(n)
        opt.zero_grad()
        running = 0.0
        for i, j in enumerate(order):
            ids, attn = model.encode([tr_s[j]], max_len=args.max_len)
            y = torch.tensor([tr_y[j]], device=model.dev0)
            _, loss = model(ids, attention_mask=attn, labels=y)
            (loss / args.grad_accum).backward()
            running += loss.item()

            if (i + 1) % args.grad_accum == 0 or (i + 1) == n:
                for g in opt.param_groups:
                    g["lr"] = lr_at(opt_step)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0)
                opt.step(); opt.zero_grad(); opt_step += 1
                if opt_step % 50 == 0:
                    rate = (i + 1) / (time.time() - t0 + 1e-9) if epoch == 1 else None
                    jl({"epoch": epoch, "opt_step": opt_step,
                        "loss": round(running / max(1, args.grad_accum), 4),
                        "lr": round(lr_at(opt_step), 8),
                        **({"seq_per_s": round((i+1)/(time.time()-t0), 2)} if epoch == 1 else {})})
                running = 0.0

        # ---- per-epoch eval on BOTH val and test (reference convention) ----
        print(f"[epoch {epoch}] training done, evaluating val + test...", flush=True)
        vm = eval_and_dump(model, val_s, val_y, args.max_len, "val", epoch, args.output_dir)
        tm = eval_and_dump(model, test_s, test_y, args.max_len, "test", epoch, args.output_dir)
        print(f"  val  ep{epoch}: R2={vm['r2']:.4f} Pearson={vm['pearson']:.4f} "
              f"Spearman={vm['spearman']:.4f} MSE={vm['mse']:.4f}", flush=True)
        print(f"  test ep{epoch}: R2={tm['r2']:.4f} Pearson={tm['pearson']:.4f} "
              f"Spearman={tm['spearman']:.4f} MSE={tm['mse']:.4f}", flush=True)
        for row in (vm, tm):
            mwriter.writerow([row["epoch"], row["split"], f"{row['r2']:.4f}",
                              f"{row['pearson']:.4f}", f"{row['spearman']:.4f}", f"{row['mse']:.4f}"])
        mcsv.flush()
        jl({"epoch_eval": epoch, "val": vm, "test": tm})

        # checkpoint best-by-val-R2
        if vm["r2"] > best["r2"]:
            best = {"r2": vm["r2"], "epoch": epoch, "test_r2": tm["r2"]}
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best.pt"))
            jl({"saved_best": True, "epoch": epoch, "val_r2": round(vm["r2"], 4),
                "test_r2": round(tm["r2"], 4)})

    print("=" * 60)
    print(f"BEST val R2={best['r2']:.4f} @ epoch {best['epoch']} "
          f"(test R2 at that epoch={best.get('test_r2', float('nan')):.4f})")
    print("=" * 60)
    json.dump(best, open(os.path.join(args.output_dir, "best_summary.json"), "w"), indent=2)
    jlog.close(); mcsv.close()


if __name__ == "__main__":
    main()
