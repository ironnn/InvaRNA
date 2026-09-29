#!/usr/bin/env python
# encoding: utf-8
"""
UNIFIED full-parameter fine-tuning for ANY embedding backbone, with ONE shared head.

Generalizes finetune_full_seqcls.py: instead of hard-coding LucaOne's own
AutoModelForSequenceClassification, it builds `UnifiedBackboneForSeqCls` = backbone
base-encoder + the single SharedSeqClsHead (unified/shared_head.py). The whole thing
(backbone + head) is trained full-parameter via the HF Trainer.

Run with the conda env that matches the backbone (see README / unified/backbones.py):
  rnafm -> threeutrlm_env             codonbert,dnabert2 -> agent2
  mrnabert -> mrnabert_env           lucaone -> lucaone        orthrus -> caduceus_env
  (evo2 is embedding-only here; full-FT is a separate ZeRO-3 track.)

Examples:
  # smoke (built-in toy regression data):
  conda run -n agent2 python finetune_unified.py --model dnabert2 --max_steps 2 --bsz 2

  # real regression:
  conda run -n lucaone python finetune_unified.py --model lucaone \
      --train_csv data/train.csv --eval_csv data/dev.csv \
      --task_type regression --output_dir ./ft_out --epochs 5 --bsz 4 --max_len 1024

CSV format: columns `seq,label`
  regression  -> label float;  multi_class/binary_class -> int;
  multi_label -> space-separated 0/1 (e.g. "0 1 1 0")
"""
import os
import sys
import argparse

# Force the torch-only backend: some envs have Keras 3 installed, which breaks
# transformers' lazy TF integration import (ValueError: Keras 3 not supported).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

# Make the repository package importable when this file is executed directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from external_models.common.backbones import get_spec, FULL_FT_KEYS
from external_models.common.model import build_finetune_model


class SeqDataset(Dataset):
    """Tokenize upfront via the backbone's encode fn (Trainer can't pass per-model kwargs)."""
    def __init__(self, seqs, labels, spec, tokenizer, max_len, task_type):
        self.encoded = [spec.encode(tokenizer, [s], max_len) for s in seqs]
        self.labels = labels
        self.task_type = task_type

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, i):
        # each encoded[i] is a dict of (1, ...) tensors -> drop the batch dim.
        # 'lengths' is a (1,) scalar count: keep it as a python int (collator re-stacks).
        item = {}
        for k, v in self.encoded[i].items():
            if k == "lengths":
                item[k] = int(v[0]) if hasattr(v, "__getitem__") else int(v)
                continue
            t = v[0] if (hasattr(v, "shape") and v.shape and v.shape[0] == 1) else v
            item[k] = t
        lab = self.labels[i]
        if self.task_type in ("regression", "multi_label", "binary_class"):
            item["labels"] = torch.tensor(lab, dtype=torch.float32)
        else:
            item["labels"] = torch.tensor(lab, dtype=torch.long)
        return item


def make_collator(pad_id, float_keys=("x",)):
    """Pad variable-length per-sample tensors to the batch max along dim 0."""
    def collate(batch):
        keys = [k for k in batch[0] if k != "labels"]
        out = {}
        for k in keys:
            vals = [b[k] for b in batch]
            # scalar metadata (e.g. orthrus 'lengths') -> 1-D tensor, no padding
            if not torch.is_tensor(vals[0]):
                out[k] = torch.tensor(vals, dtype=torch.long)
                continue
            maxlen = max(t.size(0) for t in vals)
            padval = pad_id if k == "input_ids" else 0
            padded = []
            for t in vals:
                if t.size(0) < maxlen:
                    pad_shape = (0, 0) * (t.dim() - 1) + (0, maxlen - t.size(0))
                    t = torch.nn.functional.pad(t, pad_shape, value=padval)
                padded.append(t)
            out[k] = torch.stack(padded)
        out["labels"] = torch.stack([b["labels"] for b in batch])
        return out
    return collate


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


def toy_data(task_type, num_labels, n=24):
    """Built-in smoke data: random gene seqs; GC-content as continuous/▒class target."""
    import random
    rng = random.Random(1221)
    seqs, labels = [], []
    for _ in range(n):
        L = rng.randint(60, 200)
        s = "".join(rng.choice("ACGU") for _ in range(L))
        gc = (s.count("G") + s.count("C")) / len(s)
        seqs.append(s)
        if task_type == "regression":
            labels.append(gc)
        elif task_type == "binary_class":
            labels.append(1 if gc > 0.5 else 0)
        elif task_type == "multi_label":
            labels.append([1.0 if gc > 0.5 else 0.0] + [0.0] * (num_labels - 1))
        else:  # multi_class
            labels.append(min(int(gc * num_labels), num_labels - 1))
    return seqs, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=FULL_FT_KEYS,
                    help="backbone key (run in its matching conda env)")
    ap.add_argument("--train_csv", default=None)
    ap.add_argument("--eval_csv", default=None)
    ap.add_argument("--task_type", default="regression",
                    choices=["regression", "binary_class", "multi_class", "multi_label"])
    ap.add_argument("--num_labels", type=int, default=1)
    ap.add_argument("--pooling", default="mean", choices=["mean", "cls", "max"])
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--output_dir", default="./ft_out")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--grad_ckpt", action="store_true", default=True)
    ap.add_argument("--max_steps", type=int, default=-1, help="override for smoke tests")
    args = ap.parse_args()

    # task_type / num_labels contract (head enforces too, but normalize for clarity)
    if args.task_type in ("regression", "binary_class") and args.num_labels != 1:
        print(f"[warn] task_type={args.task_type} requires num_labels=1; overriding -> 1")
        args.num_labels = 1
    if args.task_type in ("multi_class", "multi_label") and args.num_labels < 2:
        args.num_labels = 2

    spec = get_spec(args.model)

    # Single-process runs: pin to ONE GPU so the HF Trainer does NOT auto-wrap in
    # nn.DataParallel (DP scatters break our non-standard inputs: token_type_ids,
    # 6-track 'x', multimolecule heads). True multi-GPU uses DDP via torchrun, which
    # sets LOCAL_RANK/WORLD_SIZE and bypasses this guard.
    _under_torchrun = "LOCAL_RANK" in os.environ or "WORLD_SIZE" in os.environ
    if not _under_torchrun and torch.cuda.is_available() and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("[info] single-process: pinned CUDA_VISIBLE_DEVICES=0 (use torchrun for multi-GPU DDP)")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[unified-FT] backbone={args.model} (env={spec.env}, H={spec.hidden_dim}) "
          f"task={args.task_type} num_labels={args.num_labels} device={device}")

    model = build_finetune_model(
        backbone_key=args.model,
        num_labels=args.num_labels,
        task_type=args.task_type,
        pooling=args.pooling,
        device=device,
    )

    # --- FULL PARAMETER FINE-TUNING: unfreeze everything ---
    for p in model.parameters():
        p.requires_grad = True
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    n_train, n_total = model.trainable_param_report()
    print(f"[full-FT] trainable params: {n_train:,} / {n_total:,} ({100*n_train/n_total:.2f}%)")
    assert n_train == n_total, "Not all params trainable — not full fine-tuning!"

    tokenizer = spec.load_tokenizer()

    # --- data ---
    if args.train_csv:
        tr_seqs, tr_labs = read_csv(args.train_csv, args.task_type)
    else:
        print("[smoke] no --train_csv: using built-in toy data")
        tr_seqs, tr_labs = toy_data(args.task_type, args.num_labels, 24)
    train_ds = SeqDataset(tr_seqs, tr_labs, spec, tokenizer, args.max_len, args.task_type)

    eval_ds = None
    if args.eval_csv:
        ev_seqs, ev_labs = read_csv(args.eval_csv, args.task_type)
        eval_ds = SeqDataset(ev_seqs, ev_labs, spec, tokenizer, args.max_len, args.task_type)

    pad_id = 0
    if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
        pad_id = tokenizer.pad_token_id

    def metrics(eval_pred):
        logits, labels = eval_pred
        if args.task_type == "regression":
            from scipy.stats import pearsonr
            preds = np.asarray(logits).squeeze(-1)
            try:
                return {"pearson": float(pearsonr(preds, labels)[0])}
            except Exception:
                return {"mse": float(((preds - labels) ** 2).mean())}
        if args.task_type == "binary_class":
            preds = (np.asarray(logits).squeeze(-1) > 0).astype(labels.dtype)
            return {"accuracy": float((preds == labels).mean())}
        if args.task_type == "multi_label":
            preds = (np.asarray(logits) > 0).astype(labels.dtype)
            return {"acc_elem": float((preds == labels).mean())}
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=args.bf16,
        logging_steps=1,
        save_strategy="no" if args.max_steps > 0 else "epoch",
        report_to=[],
        remove_unused_columns=False,  # our collator needs token_type_ids / x / lengths
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=make_collator(pad_id),
        compute_metrics=metrics if eval_ds is not None else None,
    )

    trainer.train()
    if args.max_steps <= 0:
        save_dir = os.path.join(args.output_dir, "final")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, "unified_model.pt"))
        print(f"Done. Fine-tuned model saved to {save_dir}/unified_model.pt")
    else:
        print(f"[smoke] {args.max_steps} steps OK — full-FT path verified for '{args.model}'")


if __name__ == "__main__":
    main()
