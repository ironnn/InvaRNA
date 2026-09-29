#!/usr/bin/env python
# encoding: utf-8
"""
pretrain.py — InvaRNA backbone MLM pretraining (minimal, self-contained entry).

Replaces the old 0mymodel.py + 390-line state-spaces SequenceLightningModule + the
whole src/ framework. Built from three local modules only:
  - invarna.models           (InvaRNA model + tokenizer and Motif-MoE)
  - invarna.training.sampler (MLMDataset + mlm_getitem)
  - invarna.training.mlm_module (loss, optimizer, and scheduler)

Recipe matches the original release (step13500 weights):
  MLM 15% on ACGT only, CE ignore_index=4, AdamW lr=8e-3 wd=0.1, cosine_warmup_timm,
  bf16, grad_clip=1.0, accum=4, max_steps=50000, ckpt every 250 steps + best-by-val/loss.

Run (single node, 7 GPUs):
  python pretrain.py \
      --train_file .../train0606.fasta --val_file .../val0606.fasta \
      --devices 7 --num_nodes 1 --max_steps 50000
"""
import os
import sys
import argparse

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from datetime import datetime

# ── Compat shim: legacy .pt/.ckpt were pickled under old `caduceus.*` class paths.
# The public package lives under invarna/ and the repository root is two levels up.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
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
from invarna.training.sampler import MLMDataset
from invarna.training.mlm_module import MLMPretrainModule

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# The full publication FASTA files live with the manuscript figure assets.  The
# `assets/training_data/pretraining/` directory contains only the tiny toy files.
_DATA = os.path.join(_ROOT, "assets", "manuscript_figures")


def parse_args():
    ap = argparse.ArgumentParser(description="InvaRNA backbone MLM pretraining")
    ap.add_argument("--train_file", default=f"{_DATA}/train0606.fasta")
    ap.add_argument("--val_file",   default=f"{_DATA}/val0606.fasta")
    ap.add_argument("--config", default=os.path.join(_ROOT, "backbone", "configs", "mamba_motif_moe.yaml"),
                    help="yaml holding the backbone model.config block")
    ap.add_argument("--max_length", type=int, default=10000)
    ap.add_argument("--mlm_probability", type=float, default=0.15)
    ap.add_argument("--add_eos", action="store_true", default=True)
    ap.add_argument("--no_add_eos", dest="add_eos", action="store_false")
    # optim / schedule
    ap.add_argument("--lr", type=float, default=8e-3)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--t_initial", type=int, default=9000)
    ap.add_argument("--warmup_t", type=int, default=1000)
    # trainer
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--num_nodes", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=50000)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--val_check_interval", type=int, default=1000)
    ap.add_argument("--limit_val_batches", type=float, default=0.125)
    ap.add_argument("--ckpt_every_n_steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=2222)
    ap.add_argument("--output_dir", default=os.path.join(_HERE, "log"))
    ap.add_argument("--resume_ckpt", default=None,
                    help="path to a .ckpt to resume/continue training from "
                         "(replaces the old variants/train_step2.py step-2 flow)")
    ap.add_argument("--init_weights", default=None,
                    help="path to a .pt/.ckpt to warm-start the backbone weights from "
                         "(load_state_dict, strict=False), without resuming optimizer state")
    return ap.parse_args()


def build_model(config_path):
    cfg = OmegaConf.load(config_path)
    mc = OmegaConf.to_container(cfg.model.config, resolve=True)
    mc.pop("_target_", None)
    hf = InvaRNAConfig(**mc)
    return InvaRNAForMaskedLM(hf)


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    date_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints", date_tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    tokenizer = InvaRNATokenizer(model_max_length=args.max_length)
    print(f"[pretrain] tokenizer vocab={len(tokenizer)} pad={tokenizer.pad_token_id} "
          f"mask={tokenizer.mask_token_id}", flush=True)

    train_ds = MLMDataset(args.train_file, tokenizer, max_length=args.max_length,
                          mlm_probability=args.mlm_probability, add_eos=args.add_eos)
    val_ds = MLMDataset(args.val_file, tokenizer, max_length=args.max_length,
                        mlm_probability=args.mlm_probability, add_eos=args.add_eos)
    print(f"[pretrain] train={len(train_ds)} val={len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=False, drop_last=False)

    model = build_model(args.config)
    # optional warm-start of backbone weights (no optimizer state), e.g. from step13500.pt
    if args.init_weights is not None:
        sd = torch.load(args.init_weights, map_location="cpu")
        sd = sd.get("state_dict", sd)
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"[pretrain] init_weights loaded: missing={len(miss)} unexpected={len(unexp)}", flush=True)
    module = MLMPretrainModule(
        model, pad_token_id=tokenizer.pad_token_id,
        lr=args.lr, weight_decay=args.weight_decay,
        t_initial=args.t_initial, warmup_t=args.warmup_t,
    )

    # every-N-steps checkpoint (keep all) + best-by-val/loss
    ckpt_steps = ModelCheckpoint(
        dirpath=ckpt_dir, filename="train_step_{step}", monitor="trainer/loss",
        mode="min", save_top_k=-1, save_last=False, every_n_train_steps=args.ckpt_every_n_steps,
        auto_insert_metric_name=False)
    ckpt_best = ModelCheckpoint(
        dirpath=ckpt_dir, filename="best_val_loss", monitor="val/loss",
        mode="min", save_top_k=1, save_last=False, auto_insert_metric_name=False)
    csv_logger = CSVLogger(save_dir=args.output_dir, name="", version="")

    trainer = Trainer(
        max_steps=args.max_steps, accelerator="gpu", devices=args.devices, num_nodes=args.num_nodes,
        strategy=DDPStrategy(find_unused_parameters=False) if (args.devices > 1 or args.num_nodes > 1) else "auto",
        precision="bf16-mixed", gradient_clip_val=1.0, accumulate_grad_batches=args.grad_accum,
        val_check_interval=args.val_check_interval, limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=0, log_every_n_steps=10,
        callbacks=[ckpt_steps, ckpt_best], logger=csv_logger,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=args.resume_ckpt)
    if trainer.is_global_zero:
        print(f"[pretrain] done. best val/loss ckpt: {ckpt_best.best_model_path}", flush=True)


if __name__ == "__main__":
    main()
