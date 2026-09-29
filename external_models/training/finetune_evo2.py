#!/usr/bin/env python
# encoding: utf-8
"""
evo2 (StripedHyena 7B) full-parameter fine-tuning — custom loop.

evo2 can't use the shared finetune_unified.py path because it's not a HF model and
vortex pipeline-shards it across GPUs (HF Trainer would collapse that). This script
uses unified/evo2_ft.py (raw backbone + shared head + gradient-capable hidden tap)
with a hand-rolled training loop.

Run in an env with vortex (mamballm). Single process, NOT torchrun — the backbone is
already split across all visible GPUs by vortex.

Examples:
  # smoke (built-in toy regression):
  conda run -n mamballm python sft/finetune_evo2.py --max_steps 2 --bsz 1 --max_len 128

  # real regression:
  conda run -n mamballm python sft/finetune_evo2.py \
      --train_csv data/train.csv --task_type regression \
      --epochs 3 --bsz 1 --grad_accum 16 --max_len 1024 --output_dir ./evo2_ft

CSV: columns `seq,label` (regression=float; multi_class/binary=int; multi_label=space 0/1).
"""
import os
import sys
import argparse
os.environ.setdefault("USE_TF", "0")
import torch

LLM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(LLM_ROOT))
from external_models.common.evo2_ft import Evo2ForSeqCls


def read_csv(path, task_type):
    import csv
    seqs, labels = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            seqs.append(row["seq"].strip())
            lab = row["label"].strip()
            if task_type == "regression":
                labels.append(float(lab))
            elif task_type == "multi_label":
                labels.append([float(x) for x in lab.split()])
            else:
                labels.append(int(lab))
    return seqs, labels


def toy_data(task_type, num_labels, n=8):
    import random
    rng = random.Random(1221)
    seqs, labels = [], []
    for _ in range(n):
        L = rng.randint(60, 160)
        s = "".join(rng.choice("ACGT") for _ in range(L))
        gc = (s.count("G") + s.count("C")) / len(s)
        seqs.append(s)
        if task_type == "regression":
            labels.append(gc)
        elif task_type == "binary_class":
            labels.append(1 if gc > 0.5 else 0)
        elif task_type == "multi_label":
            labels.append([1.0 if gc > 0.5 else 0.0] + [0.0] * (num_labels - 1))
        else:
            labels.append(min(int(gc * num_labels), num_labels - 1))
    return seqs, labels


def label_tensor(labs, task_type, device):
    if task_type in ("regression", "binary_class", "multi_label"):
        return torch.tensor(labs, dtype=torch.float32, device=device)
    return torch.tensor(labs, dtype=torch.long, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default=None)
    ap.add_argument("--task_type", default="regression",
                    choices=["regression", "binary_class", "multi_class", "multi_label"])
    ap.add_argument("--num_labels", type=int, default=1)
    ap.add_argument("--pooling", default="mean", choices=["mean", "cls", "max"])
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--output_dir", default="./evo2_ft")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_steps", type=int, default=-1)
    args = ap.parse_args()

    if args.task_type in ("regression", "binary_class") and args.num_labels != 1:
        args.num_labels = 1
    if args.task_type in ("multi_class", "multi_label") and args.num_labels < 2:
        args.num_labels = 2

    print(f"[evo2-FT] task={args.task_type} num_labels={args.num_labels} "
          f"(vortex pipeline-sharded across {torch.cuda.device_count()} GPUs)", flush=True)

    model = Evo2ForSeqCls(num_labels=args.num_labels, task_type=args.task_type, pooling=args.pooling)
    model.train()
    n_train, n_total = model.trainable_param_report()
    print(f"[full-FT] trainable params: {n_train:,} / {n_total:,} ({100*n_train/n_total:.2f}%)", flush=True)
    assert n_train == n_total, "Not all params trainable!"

    if args.train_csv:
        seqs, labs = read_csv(args.train_csv, args.task_type)
    else:
        print("[smoke] no --train_csv: using built-in toy data", flush=True)
        seqs, labs = toy_data(args.task_type, args.num_labels, 8)

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    n = len(seqs)
    steps_per_epoch = max(1, n // args.bsz)
    total_steps = args.max_steps if args.max_steps > 0 else int(args.epochs * steps_per_epoch)
    print(f"[train] {n} samples, ~{total_steps} optimizer steps", flush=True)

    step = 0
    opt.zero_grad()
    done = False
    epoch = 0
    while not done:
        epoch += 1
        order = list(range(n))
        for bi in range(0, n, args.bsz):
            idx = order[bi:bi + args.bsz]
            bseqs = [seqs[i] for i in idx]
            blabs = [labs[i] for i in idx]
            input_ids, attn = model.encode(bseqs, max_len=args.max_len)
            y = label_tensor(blabs, args.task_type, model.dev0)
            _, loss = model(input_ids, attention_mask=attn, labels=y)
            (loss / args.grad_accum).backward()

            if ((bi // args.bsz) + 1) % args.grad_accum == 0 or (bi + args.bsz) >= n:
                opt.step()
                opt.zero_grad()
            step += 1
            print(f"  epoch {epoch} step {step}/{total_steps} loss={loss.item():.4f}", flush=True)
            if step >= total_steps:
                done = True
                break

    if args.max_steps <= 0:
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, "evo2_finetuned.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Done. Saved to {save_path}", flush=True)
    else:
        print(f"[smoke] {total_steps} steps OK — evo2 full-FT path verified", flush=True)


if __name__ == "__main__":
    main()
