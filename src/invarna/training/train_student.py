#!/usr/bin/env python
# encoding: utf-8
"""
sft.py — InvaRNA backbone supervised fine-tuning (regression head) on aligned mRNA.

Loads the packaged backbone, adds the checkpoint-compatible regression head,
full fine-tunes (backbone + head) with differential LR, and
dumps per-epoch val/test R2 CSVs.

Alignment (identical to the selected finetune_stage2.py): short 5' UTRs are padded
toward CDS@1000; overlength 5' UTRs are not left-truncated; total length 10000; N->pad.

Head (--head):
  mambarc (default) : RC-CNN head + regressor, verbatim from InvaRNA MambaRCHeadStage2.
                      Mounted FLAT (rc_cnn_head.* / regressor.*) so the saved state_dict is
                      key-identical to the released assets/checkpoints/te_student/*.ckpt.
                      loads both with the same stage2 loader.
  mlp               : masked mean-pool + MLP (own key namespace; not cross-loadable).

Data: --train/--val/--test parquet, columns configurable via --seq_col/--label_col plus
utr5/cds size columns for alignment. Default matches pure_te_bench (mrna, utr5_size, mean_te).

Per-epoch: val+test all_gather -> sklearn r2 -> {output_dir}/{split}_ep{N}_r2_{r2}.csv;
best by val_r2_global; metrics.csv via CSVLogger. Differential LR backbone 5e-6 / head 1e-3.

Example:
  python sft.py \
      --train .../human_train_wt.parquet --val .../human_val_wt.parquet \
      --test .../human_test_wt.parquet --epochs 30 --devices 7 \
      --backbone_lr 5e-6 --head_lr 1e-3 --output_dir ./sft_out
"""
import os
import sys
import argparse
import yaml

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from omegaconf import OmegaConf

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))

# ── compat shim + repo backbone
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


# ─────────────────────────────────────────────────────────────────────────────
# Alignment + dataset (verbatim from finetune_stage2.py)
# ─────────────────────────────────────────────────────────────────────────────
def align_and_pad_sequence(sequence, utr5_length):
    """CDS-anchored padding: pin start codon at FIXED_CDS_START, pad/clip to TOTAL_LENGTH."""
    sequence = str(sequence)
    left_padding = max(FIXED_CDS_START - int(utr5_length), 0)
    new_sequence = "N" * left_padding + sequence
    new_sequence = new_sequence[:TOTAL_LENGTH].ljust(TOTAL_LENGTH, "N")
    return new_sequence


class RNAPaddedSeqDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.unk_id = tokenizer.convert_tokens_to_ids("N")
        self.pad_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_ids = self.tokenizer.encode(self.sequences[idx], add_special_tokens=False)
        input_ids = [self.pad_id if t == self.unk_id else t for t in input_ids]
        return {"input_ids": input_ids, "label": self.labels[idx]}


def collate_fn(batch):
    input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    PAD_ID, N_ID = 4, 11
    attention_mask = ((input_ids != PAD_ID) & (input_ids != N_ID)).long()
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ─────────────────────────────────────────────────────────────────────────────
# Heads
#
# The default "mambarc" head is mounted FLAT on the LightningModule as
# self.rc_cnn_head / self.regressor (NOT nested under self.head), so that the
# saved state_dict keys are byte-identical to InvaRNA's MambaRCHeadStage2 /
# released assets/checkpoints/student/*.ckpt:  backbone.* + rc_cnn_head.* + regressor.*
# This means the evaluation inference API can load this repo's SFT output and released
# student ckpts with the SAME stage2 loader (no prefix juggling).
#
# Alternative heads (e.g. "mlp") are mounted under self.head — they have their own
# key namespace and are not cross-loadable with the released stage2 ckpts.
# ─────────────────────────────────────────────────────────────────────────────
def build_mambarc_head(d_model=512):
    """RC-CNN head + regressor — verbatim from InvaRNA MambaRCHeadStage2.
    Returns (rc_cnn_head, regressor) mounted flat so keys match the released ckpt."""
    rc_cnn_head = nn.Sequential(
        nn.Conv1d(d_model, 256, kernel_size=5, padding=2, groups=2),
        nn.BatchNorm1d(256), nn.SiLU(), nn.MaxPool1d(2),
        nn.Conv1d(256, 128, kernel_size=5, padding=2, groups=2),
        nn.BatchNorm1d(128), nn.SiLU(), nn.MaxPool1d(2),
        nn.Conv1d(128, 128, kernel_size=5, padding=2, groups=2),
        nn.BatchNorm1d(128), nn.SiLU(), nn.MaxPool1d(2),
        nn.AdaptiveAvgPool1d(64),
    )
    regressor = nn.Sequential(
        nn.Flatten(), nn.Dropout(0.3),
        nn.Linear(128 * 64, 256), nn.SiLU(), nn.Dropout(0.3),
        nn.Linear(256, 1),
    )
    return rc_cnn_head, regressor


class MeanMLPHead(nn.Module):
    """Lightweight alternative head: masked mean-pool + MLP. Mounted under self.head;
    NOT key-compatible with the released stage2 ckpts (own namespace)."""
    def __init__(self, d_model=512, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.SiLU(), nn.Dropout(0.3), nn.Linear(hidden, 1))

    def forward(self, hidden_states):
        pooled = hidden_states.mean(dim=1)
        return self.net(pooled)


HEAD_CHOICES = ["mambarc", "mlp"]


# ─────────────────────────────────────────────────────────────────────────────
# Lightning module
# ─────────────────────────────────────────────────────────────────────────────
class InvaRNASFTModule(LightningModule):
    def __init__(self, backbone_model, head_name, save_dir, head_lr=1e-3, backbone_lr=5e-6,
                 real_df=None, syn_pool_df=None, tokenizer=None, dynamic_sampling=False,
                 epoch_syn_sample_size=40000, real_copy_factor=2, batch_size=12,
                 num_workers=4):
        super().__init__()
        self.backbone = backbone_model
        self.head_name = head_name
        if head_name == "mambarc":
            # flat layout -> keys identical to MambaRCHeadStage2 / student/*.ckpt
            self.rc_cnn_head, self.regressor = build_mambarc_head(d_model=512)
            self.head = None
        else:
            self.head = MeanMLPHead(d_model=512)
        self.save_dir = save_dir
        self.head_lr = head_lr
        self.backbone_lr = backbone_lr
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.real_df = real_df
        self.syn_pool_df = syn_pool_df
        self.tokenizer = tokenizer
        self.dynamic_sampling = dynamic_sampling
        self.epoch_syn_sample_size = int(epoch_syn_sample_size)
        self.real_copy_factor = int(real_copy_factor)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)

    def _apply_head(self, hidden_states):
        if self.head_name == "mambarc":
            x = hidden_states.permute(0, 2, 1)
            x = self.rc_cnn_head(x)
            return self.regressor(x)
        return self.head(hidden_states)

    def _head_parameters(self):
        if self.head_name == "mambarc":
            return list(self.rc_cnn_head.parameters()) + list(self.regressor.parameters())
        return list(self.head.parameters())

    def forward(self, input_ids, attention_mask=None):
        hidden_states, *_ = self.backbone(input_ids, output_hidden_states=False, return_dict=False)
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return self._apply_head(hidden_states)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch["input_ids"], batch["attention_mask"])
        loss = nn.MSELoss()(y_hat.view(-1), batch["labels"].view(-1))
        self.log("train_loss", loss, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        y = batch["labels"].view(-1)
        y_hat = self(batch["input_ids"], batch["attention_mask"]).view(-1)
        loss = nn.MSELoss()(y_hat, y)
        rec = {"pred": y_hat.detach(), "target": y.detach()}
        if dataloader_idx == 0:
            self.log("val_loss", loss, sync_dist=True, add_dataloader_idx=False, prog_bar=True)
            self.validation_step_outputs.append(rec)
        else:
            self.test_step_outputs.append(rec)
        return loss

    def on_validation_epoch_end(self):
        def process(outputs, split):
            if not outputs:
                return
            preds = self.all_gather(torch.cat([o["pred"] for o in outputs])).view(-1).float().cpu().numpy()
            targs = self.all_gather(torch.cat([o["target"] for o in outputs])).view(-1).float().cpu().numpy()
            g_r2 = r2_score(targs, preds)
            self.log(f"{split}_r2_global", g_r2, sync_dist=False)
            if self.trainer.is_global_zero:
                print(f"\n[Global Sklearn] {split} R2: {g_r2:.4f}", flush=True)
                os.makedirs(self.save_dir, exist_ok=True)
                pd.DataFrame({"pred": preds, "target": targs}).to_csv(
                    f"{self.save_dir}/{split}_ep{self.current_epoch}_r2_{g_r2:.4f}.csv", index=False)
        process(self.validation_step_outputs, "val")
        process(self.test_step_outputs, "test")
        self.validation_step_outputs.clear()
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {"params": self.backbone.parameters(), "lr": self.backbone_lr},
            {"params": self._head_parameters(), "lr": self.head_lr},
        ], weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_r2_global",
                                 "interval": "epoch", "frequency": 1, "strict": False}}

    def train_dataloader(self):
        if not self.dynamic_sampling:
            raise RuntimeError("module-owned dataloader is only used for dynamic sampling")
        pieces = []
        if self.real_df is not None and not self.real_df.empty:
            pieces.append(pd.concat([self.real_df] * self.real_copy_factor, ignore_index=True))
        sample_size = min(self.epoch_syn_sample_size, len(self.syn_pool_df))
        if sample_size:
            pieces.append(self.syn_pool_df.sample(n=sample_size, replace=False))
        frame = pd.concat(pieces, ignore_index=True).sample(frac=1.0).reset_index(drop=True)
        sequences = [align_and_pad_sequence(seq, utr5) for seq, utr5 in zip(frame["mrna"], frame["utr5_size"])]
        print(f"[epoch] WT={len(self.real_df)} x{self.real_copy_factor}; synthetic={sample_size}; total={len(frame)}", flush=True)
        return DataLoader(
            RNAPaddedSeqDataset(sequences, frame["final_label"].astype("float32").tolist(), self.tokenizer),
            batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn,
            num_workers=self.num_workers,
        )


# ─────────────────────────────────────────────────────────────────────────────
def load_split(path, seq_col, utr5_col, label_col):
    df = pd.read_parquet(path)
    seqs = [align_and_pad_sequence(s, u) for s, u in zip(df[seq_col], df[utr5_col])]
    labels = df[label_col].astype("float32").tolist()
    return seqs, labels


def build_backbone(config_path, weights_path):
    mc = OmegaConf.to_container(OmegaConf.load(config_path).model.config, resolve=True)
    mc.pop("_target_", None)
    model = InvaRNAForMaskedLM(InvaRNAConfig(**mc))
    sd = torch.load(weights_path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[sft] backbone load: missing={len(miss)} unexpected={len(unexp)}", flush=True)
    return model.invaRNA


def main():
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--student-config")
    known, _ = config_probe.parse_known_args()
    student = {}
    if known.student_config:
        with open(known.student_config, encoding="utf-8") as handle:
            student = yaml.safe_load(handle) or {}
    training = student.get("training", {})
    train_default = student.get("train_data")
    if isinstance(train_default, str):
        train_default = [train_default]
    configured_seed = student.get("seed")
    if not isinstance(configured_seed, int):
        configured_seed = None

    ap = argparse.ArgumentParser(description="InvaRNA backbone SFT (regression)")
    _D = os.path.join(_HERE, "data")
    ap.add_argument("--student-config", default=known.student_config,
                    help="Student experiment YAML; explicit CLI options override it")
    ap.add_argument("--train", nargs="+",
                    default=train_default or [os.path.join(_D, "human_train_wt.parquet"),
                                              os.path.join(_D, "mouse_train_wt.parquet")],
                    help="train parquet(s), concatenated")
    ap.add_argument("--val", default=student.get("validation_data", os.path.join(_D, "human_val_wt.parquet")))
    ap.add_argument("--test", default=student.get("test_data", os.path.join(_D, "human_test_wt.parquet")))
    ap.add_argument("--seq_col", default="mrna")
    ap.add_argument("--utr5_col", default="utr5_size")
    ap.add_argument("--label_col", default=student.get("label_col", "mean_te"))
    ap.add_argument("--evaluation_label_col", default=student.get("evaluation_label_col", "mean_te"),
                    help="Measured label used for validation/test metrics")
    ap.add_argument("--head", default="mambarc", choices=HEAD_CHOICES)
    ap.add_argument("--epochs", type=int, default=training.get("max_epochs", 30))
    ap.add_argument("--bsz", type=int, default=training.get("batch_size", 12))
    ap.add_argument("--grad_accum", type=int, default=training.get("accumulate_grad_batches", 4))
    ap.add_argument("--head_lr", type=float, default=training.get("head_lr", 1e-3))
    ap.add_argument("--backbone_lr", type=float, default=training.get("backbone_lr", 5e-6))
    ap.add_argument("--devices", type=int, default=training.get("devices", 1))
    ap.add_argument("--num_workers", type=int, default=training.get("num_workers", 4))
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--init_weights", default=training.get("backbone_weights", DEFAULT_WEIGHTS))
    ap.add_argument("--output_dir", default="./sft_out")
    ap.add_argument("--seed", type=int, default=configured_seed)
    ap.add_argument("--precision", default=training.get("precision", "bf16-mixed"))
    ap.add_argument("--matmul_precision", default=training.get("matmul_precision", "high"))
    ap.add_argument("--max_train", type=int, default=-1)
    ap.add_argument("--dynamic_sampling", action=argparse.BooleanOptionalAction,
                    default=training.get("dynamic_sampling", False),
                    help="Resample the synthetic pool without replacement every epoch")
    ap.add_argument("--epoch_syn_sample_size", type=int,
                    default=training.get("epoch_synthetic_sample_size", 40000))
    ap.add_argument("--real_copy_factor", type=int, default=training.get("real_copy_factor", 2))
    args = ap.parse_args()

    if args.seed is None:
        ap.error("a confirmed integer --seed is required; selected-run YAMLs intentionally mark it NEEDS_AUTHOR_CONFIRMATION")

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(args.matmul_precision)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = InvaRNATokenizer(model_max_length=TOTAL_LENGTH)

    tr_seqs, tr_labs = [], []
    real_df = syn_pool_df = None
    if args.dynamic_sampling:
        frames = [pd.read_parquet(path) for path in args.train]
        frame = pd.concat(frames, ignore_index=True)
        needed = {args.seq_col, args.utr5_col, args.label_col, "mean_te", "transcript_id"}
        missing = needed - set(frame.columns)
        if missing:
            raise ValueError(f"dynamic training table missing columns: {sorted(missing)}")
        if "variant_source" in frame:
            is_wt = frame["variant_source"].eq("wt")
        else:
            is_wt = frame["transcript_id"].astype(str).str.endswith("_mut0")
        if not is_wt.any():
            raise ValueError("dynamic training requires WT rows marked variant_source=wt or *_mut0")
        frame = frame.rename(columns={args.seq_col: "mrna", args.utr5_col: "utr5_size"})
        frame["final_label"] = frame[args.label_col]
        frame.loc[is_wt, "final_label"] = frame.loc[is_wt, "mean_te"]
        real_df, syn_pool_df = frame[is_wt].copy(), frame[~is_wt].copy()
        if args.max_train > 0:
            syn_pool_df = syn_pool_df.head(args.max_train)
    else:
        for p in args.train:
            s, l = load_split(p, args.seq_col, args.utr5_col, args.label_col)
            tr_seqs += s; tr_labs += l
        if args.max_train > 0:
            tr_seqs, tr_labs = tr_seqs[:args.max_train], tr_labs[:args.max_train]
    val_seqs, val_labs = load_split(
        args.val, args.seq_col, args.utr5_col, args.evaluation_label_col
    )
    train_count = len(real_df) + len(syn_pool_df) if args.dynamic_sampling else len(tr_seqs)
    print(f"[sft] train_pool={train_count} val={len(val_seqs)} head={args.head} dynamic={args.dynamic_sampling}", flush=True)

    train_loader = None
    if not args.dynamic_sampling:
        train_loader = DataLoader(RNAPaddedSeqDataset(tr_seqs, tr_labs, tokenizer),
                                  batch_size=args.bsz, shuffle=True, collate_fn=collate_fn,
                                  num_workers=args.num_workers)
    val_loaders = [DataLoader(RNAPaddedSeqDataset(val_seqs, val_labs, tokenizer),
                              batch_size=args.bsz, shuffle=False, collate_fn=collate_fn,
                              num_workers=args.num_workers)]
    if args.test:
        te_seqs, te_labs = load_split(
            args.test, args.seq_col, args.utr5_col, args.evaluation_label_col
        )
        val_loaders.append(DataLoader(RNAPaddedSeqDataset(te_seqs, te_labs, tokenizer),
                                      batch_size=args.bsz, shuffle=False, collate_fn=collate_fn,
                                      num_workers=args.num_workers))

    bb = build_backbone(args.config, args.init_weights)
    module = InvaRNASFTModule(bb, args.head, save_dir=args.output_dir,
                              head_lr=args.head_lr, backbone_lr=args.backbone_lr,
                              real_df=real_df, syn_pool_df=syn_pool_df, tokenizer=tokenizer,
                              dynamic_sampling=args.dynamic_sampling,
                              epoch_syn_sample_size=args.epoch_syn_sample_size,
                              real_copy_factor=args.real_copy_factor,
                              batch_size=args.bsz, num_workers=args.num_workers)

    ckpt = ModelCheckpoint(dirpath=args.output_dir, filename="best-{epoch:02d}-{val_r2_global:.4f}",
                           monitor="val_r2_global", mode="max", save_top_k=1, save_weights_only=True)
    trainer = Trainer(
        max_epochs=args.epochs, accelerator="gpu", devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=False) if args.devices > 1 else "auto",
        precision=args.precision, log_every_n_steps=5, accumulate_grad_batches=args.grad_accum,
        callbacks=[ckpt], logger=CSVLogger(save_dir=args.output_dir, name="", version=""),
        num_sanity_val_steps=0, reload_dataloaders_every_n_epochs=1 if args.dynamic_sampling else 0,
    )
    if args.dynamic_sampling:
        trainer.fit(module, val_dataloaders=val_loaders)
    else:
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loaders)
    if trainer.is_global_zero:
        print(f"[sft] best val_r2_global={ckpt.best_model_score} @ {ckpt.best_model_path}", flush=True)


if __name__ == "__main__":
    main()
