#!/usr/bin/env python
# encoding: utf-8
"""
(b) FULL-PARAMETER FINE-TUNING of LucaOne for gene (DNA/RNA) sequence classification.

Loads the LOCAL converted HF checkpoint (no huggingface.co), unfreezes ALL
parameters (LucaOne encoder + classifier head), and trains with the HF Trainer.

This is true 全参微调: the entire 20-layer / 2560-hidden LucaOne backbone receives
gradients, unlike LucaOneTasks (which trains only a head on frozen embeddings).

Example (toy smoke test is built in if --train_csv is omitted):
  conda run -n lucaone python finetune_full_seqcls.py \
      --model_dir assets/checkpoints/external/pretrained/models/lucaone_hf_ckpts/LucaOne-gene-step36.8M \
      --train_csv mydata/train.csv --eval_csv mydata/dev.csv \
      --num_labels 4 --task_type multi_class --seq_type gene \
      --output_dir ./ft_out --epochs 5 --bsz 4 --grad_accum 8 --max_len 1024

CSV format: columns `seq,label` (label = int class index for multi_class/binary_class,
float for regression; for multi_label use space-separated 0/1, e.g. "0 1 1 0").
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, Trainer, TrainingArguments,
)
import lucaone  # noqa: F401  triggers Auto* registration


class SeqDataset(Dataset):
    """Tokenizes upfront with seq_type (HF Trainer can't pass seq_type itself)."""
    def __init__(self, seqs, labels, tokenizer, seq_type, max_len, task_type):
        self.enc = []
        for s in seqs:
            e = tokenizer(
                s, seq_type=seq_type, add_special_tokens=True,
                truncation=True, max_length=max_len,
            )
            self.enc.append(e)
        self.labels = labels
        self.task_type = task_type

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, i):
        item = {k: torch.tensor(v) for k, v in self.enc[i].items()}
        lab = self.labels[i]
        # binary_class -> BCEWithLogitsLoss (1 logit, float);
        # regression -> MSE/L1 (float); multi_label -> BCE (float vector);
        # multi_class -> CrossEntropy (long).
        if self.task_type in ("regression", "multi_label", "binary_class"):
            item["labels"] = torch.tensor(lab, dtype=torch.float32)
        else:
            item["labels"] = torch.tensor(lab, dtype=torch.long)
        return item


def make_collator(pad_id):
    def collate(batch):
        keys = [k for k in batch[0] if k != "labels"]
        maxlen = max(b["input_ids"].size(0) for b in batch)
        out = {}
        for k in keys:
            padval = pad_id if k == "input_ids" else 0
            out[k] = torch.stack([
                torch.nn.functional.pad(b[k], (0, maxlen - b[k].size(0)), value=padval)
                for b in batch
            ])
        out["labels"] = torch.stack([b["labels"] for b in batch])
        return out
    return collate


def read_csv(path, task_type):
    import csv
    seqs, labels = [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            seqs.append(row["seq"].strip())
            lab = row["label"].strip()
            if task_type == "regression":
                labels.append(float(lab))
            elif task_type == "multi_label":
                labels.append([float(x) for x in lab.split()])
            else:
                labels.append(int(lab))
    return seqs, labels


def toy_data(n=24):
    """Built-in smoke-test data: random gene seqs, 2 classes by GC-ish rule."""
    import random
    rng = random.Random(1221)
    seqs, labels = [], []
    for _ in range(n):
        L = rng.randint(60, 200)
        s = "".join(rng.choice("ACGU") for _ in range(L))
        lab = 1 if (s.count("G") + s.count("C")) / len(s) > 0.5 else 0
        seqs.append(s)
        labels.append(lab)
    return seqs, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--train_csv", default=None)
    ap.add_argument("--eval_csv", default=None)
    ap.add_argument("--seq_type", default="gene", choices=["gene", "prot"])
    ap.add_argument("--task_level", default="seq_level", choices=["seq_level", "token_level"])
    ap.add_argument("--task_type", default="multi_class",
                    choices=["multi_class", "binary_class", "regression", "multi_label"])
    ap.add_argument("--num_labels", type=int, default=2)
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

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    ModelCls = (AutoModelForTokenClassification if args.task_level == "token_level"
                else AutoModelForSequenceClassification)

    # task-type contract enforced by modeling_lucaone:
    #   multi_class  -> CrossEntropy, num_labels = #classes (>=2), long labels
    #   binary_class -> BCEWithLogits, num_labels = 1, float labels
    #   regression   -> MSE/L1,       num_labels = 1, float labels
    #   multi_label  -> BCEWithLogits, num_labels = #labels, float 0/1 vectors
    if args.task_type in ("binary_class", "regression") and args.num_labels != 1:
        print(f"[warn] task_type={args.task_type} requires num_labels=1; overriding {args.num_labels} -> 1")
        args.num_labels = 1

    model = ModelCls.from_pretrained(
        args.model_dir,
        task_level=args.task_level,
        task_type=args.task_type,
        classifier_num_labels=args.num_labels,
        trust_remote_code=True,
    )

    # --- FULL PARAMETER FINE-TUNING: unfreeze everything ---
    for p in model.parameters():
        p.requires_grad = True
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[full-FT] trainable params: {n_train:,} / {n_total:,} "
          f"({100*n_train/n_total:.2f}%)")
    assert n_train == n_total, "Not all params trainable — not full fine-tuning!"

    # --- data ---
    if args.train_csv:
        tr_seqs, tr_labs = read_csv(args.train_csv, args.task_type)
    else:
        print("[smoke] no --train_csv: using built-in toy gene data")
        tr_seqs, tr_labs = toy_data(24)
    train_ds = SeqDataset(tr_seqs, tr_labs, tok, args.seq_type, args.max_len, args.task_type)

    eval_ds = None
    if args.eval_csv:
        ev_seqs, ev_labs = read_csv(args.eval_csv, args.task_type)
        eval_ds = SeqDataset(ev_seqs, ev_labs, tok, args.seq_type, args.max_len, args.task_type)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    def metrics(eval_pred):
        logits, labels = eval_pred
        if args.task_type == "regression":
            from scipy.stats import pearsonr
            preds = logits.squeeze(-1)
            return {"pearson": float(pearsonr(preds, labels)[0])}
        if args.task_type == "binary_class":
            preds = (logits.squeeze(-1) > 0).astype(labels.dtype)
            return {"accuracy": float((preds == labels).mean())}
        if args.task_type == "multi_label":
            preds = (logits > 0).astype(labels.dtype)
            return {"f1_micro": float((preds == labels).mean())}
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
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        report_to=[],
        remove_unused_columns=False,   # our collator needs token_type_ids etc.
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
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tok.save_pretrained(os.path.join(args.output_dir, "final"))
    print(f"Done. Fine-tuned model saved to {os.path.join(args.output_dir, 'final')}")


if __name__ == "__main__":
    main()
