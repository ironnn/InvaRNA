#!/usr/bin/env python
# encoding: utf-8
"""
lucaone full-parameter fine-tuning on the pure_te_bench TE dataset (Lightning DDP).

Region : whole mRNA from 5' end, read 5'->3'.
MaxLen : 1280 tokens (LucaOne convention; right-truncate, keep 5' start).
Env    : lucaone

Harness follows InvaRNA/train/finetune_stage2.py:
  - PyTorch-Lightning + DDPStrategy(find_unused_parameters=False), precision bf16-mixed
  - label=mean_te, loss=MSELoss; head = native if backbone ships one else shared
  - per-epoch val+test eval: all_gather -> sklearn r2_score -> dump
        {save_dir}/{split}_ep{epoch}_r2_{r2:.4f}.csv   (pred,target)
  - ModelCheckpoint(monitor=val_r2_global, mode=max); metrics.csv via CSVLogger

Train: human_train + mouse_train (20350)   Val: human_val   Test: human_test

Run (multi-GPU DDP):
  conda run -n lucaone python sft/train_te_lucaone.py \
      --devices 4 --epochs 4 --bsz 1 --grad_accum 16 --output_dir ./te_lucaone_out
Single GPU:
  conda run -n lucaone python sft/train_te_lucaone.py --devices 1 ...
"""
import os, sys, argparse
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger

LLM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(LLM_ROOT))
from external_models.common.backbones import get_spec
from external_models.common.te_lightning import TELightningModule

MODEL_KEY = "lucaone"
DEFAULT_MAX_LEN = 1280


def extract_region(mrna, utr5, cds, utr3):
    """Reads the whole mRNA from the 5' end, 5'->3'."""
    return mrna


def load_split(path):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["mrna", "utr5_size", "cds_size", "utr3_size", "mean_te"]).to_pandas()
    seqs, labs = [], []
    for _, r in t.iterrows():
        s = extract_region(r["mrna"], int(r["utr5_size"]), int(r["cds_size"]), int(r["utr3_size"]))
        if len(s) == 0:
            continue
        seqs.append(s); labs.append(float(r["mean_te"]))
    return seqs, np.asarray(labs, dtype=np.float64)


class SeqDS(Dataset):
    def __init__(self, seqs, labs, spec, tok, max_len):
        self.enc = [spec.encode(tok, [s], max_len) for s in seqs]
        self.labs = labs
    def __len__(self): return len(self.enc)
    def __getitem__(self, i):
        item = {}
        for k, v in self.enc[i].items():
            if k == "lengths":
                item[k] = int(v[0]) if hasattr(v, "__getitem__") else int(v); continue
            item[k] = v[0] if (hasattr(v, "shape") and v.shape and v.shape[0] == 1) else v
        item["labels"] = torch.tensor(self.labs[i], dtype=torch.float32)
        return item


def make_collator(pad_id):
    def collate(batch):
        out = {}
        for k in [k for k in batch[0] if k != "labels"]:
            vals = [b[k] for b in batch]
            if not torch.is_tensor(vals[0]):
                out[k] = torch.tensor(vals, dtype=torch.long); continue
            maxlen = max(t.size(0) for t in vals)
            padval = pad_id if k == "input_ids" else 0
            padded = []
            for t in vals:
                if t.size(0) < maxlen:
                    t = torch.nn.functional.pad(t, (0, 0) * (t.dim() - 1) + (0, maxlen - t.size(0)), value=padval)
                padded.append(t)
            out[k] = torch.stack(padded)
        out["labels"] = torch.stack([b["labels"] for b in batch])
        return out
    return collate


class TEDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, spec, max_len, bsz, num_workers=4, max_train=-1):
        super().__init__()
        self.data_dir = data_dir; self.spec = spec; self.max_len = max_len
        self.bsz = bsz; self.num_workers = num_workers; self.max_train = max_train
        self.tok = spec.load_tokenizer()
        pad_id = getattr(self.tok, "pad_token_id", 0) or 0
        self.collate = make_collator(pad_id)

    def setup(self, stage=None):
        d = self.data_dir
        h_s, h_y = load_split(os.path.join(d, "human_train_wt.parquet"))
        m_s, m_y = load_split(os.path.join(d, "mouse_train_wt.parquet"))
        tr_s = h_s + m_s; tr_y = np.concatenate([h_y, m_y])
        val_s, val_y = load_split(os.path.join(d, "human_val_wt.parquet"))
        test_s, test_y = load_split(os.path.join(d, "human_test_wt.parquet"))
        if self.max_train > 0:
            tr_s, tr_y = tr_s[:self.max_train], tr_y[:self.max_train]
            val_s, val_y, test_s, test_y = val_s[:64], val_y[:64], test_s[:64], test_y[:64]
        self.tr = SeqDS(tr_s, tr_y, self.spec, self.tok, self.max_len)
        self.val = SeqDS(val_s, val_y, self.spec, self.tok, self.max_len)
        self.test = SeqDS(test_s, test_y, self.spec, self.tok, self.max_len)
        print(f"[data] train={len(self.tr)} val={len(self.val)} test={len(self.test)}", flush=True)

    def train_dataloader(self):
        return DataLoader(self.tr, batch_size=self.bsz, shuffle=True,
                          collate_fn=self.collate, num_workers=self.num_workers, drop_last=False)
    def val_dataloader(self):
        v = DataLoader(self.val, batch_size=self.bsz, shuffle=False, collate_fn=self.collate, num_workers=self.num_workers)
        t = DataLoader(self.test, batch_size=self.bsz, shuffle=False, collate_fn=self.collate, num_workers=self.num_workers)
        return [v, t]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="assets/training_data/common_te_splits")
    ap.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_ckpt", action="store_true", default=True)
    ap.add_argument("--devices", type=int, default=4, help="number of GPUs (DDP if >1)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--output_dir", default="./te_lucaone_out")
    ap.add_argument("--seed", type=int, default=2222)
    ap.add_argument("--max_train", type=int, default=-1)
    args = ap.parse_args()

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    os.makedirs(args.output_dir, exist_ok=True)
    spec = get_spec(MODEL_KEY)
    print(f"[TE-{MODEL_KEY}] region=mRNA(5') max_len={args.max_len}tok env={spec.env} devices={args.devices}", flush=True)

    dm = TEDataModule(args.data_dir, spec, args.max_len, args.bsz, args.num_workers, args.max_train)
    model = TELightningModule(MODEL_KEY, save_dir=args.output_dir, lr=args.lr,
                              weight_decay=args.weight_decay, grad_ckpt=args.grad_ckpt)

    ckpt = ModelCheckpoint(dirpath=args.output_dir, filename="best-{epoch:02d}-{val_r2_global:.4f}",
                           monitor="val_r2_global", mode="max", save_top_k=1, save_weights_only=True)
    csv_logger = CSVLogger(save_dir=args.output_dir, name="", version="")
    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator="gpu", devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=False) if args.devices > 1 else "auto",
        precision="bf16-mixed", log_every_n_steps=10,
        callbacks=[ckpt], logger=csv_logger,
        accumulate_grad_batches=args.grad_accum, use_distributed_sampler=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=dm)
    if trainer.is_global_zero:
        print(f"BEST val_r2_global = {ckpt.best_model_score} @ {ckpt.best_model_path}", flush=True)


if __name__ == "__main__":
    main()
