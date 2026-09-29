"""
Shared PyTorch-Lightning core for TE full-parameter fine-tuning of the unified
backbones (everything except evo2, which uses sft/train_te_evo2.py).

Follows InvaRNA/train/finetune_stage2.py conventions:
  - label = mean_te (regression), loss = MSELoss
  - two val dataloaders: idx 0 = val, idx 1 = test
  - on_validation_epoch_end: all_gather preds/targets across ranks, compute
    sklearn r2_score, log val_r2_global / test_r2_global, and (rank 0 only) dump
    {save_dir}/{split}_ep{epoch}_r2_{r2:.4f}.csv  (columns pred,target)
  - Trainer(strategy=DDPStrategy(find_unused_parameters=False), precision="bf16-mixed")
  - ModelCheckpoint(monitor="val_r2_global", mode="max")

Head policy (unchanged): native regression head when the backbone ships one
(codonbert/rnafm/dnabert2/mrnabert/lucaone), else the shared head (orthrus).
We reuse unified.model.build_finetune_model so the model is identical to the
single-GPU / evo2 path — only the training harness differs.
"""
import os
import csv
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

from external_models.common.backbones import get_spec
from external_models.common.model import build_finetune_model


class TELightningModule(pl.LightningModule):
    def __init__(self, backbone_key, save_dir, lr=1e-5, weight_decay=0.01,
                 warmup_epochs=0, grad_ckpt=True):
        super().__init__()
        self.save_hyperparameters(ignore=[])
        self.backbone_key = backbone_key
        self.save_dir = save_dir
        self.lr = lr
        self.weight_decay = weight_decay

        # build_finetune_model returns backbone + (native|shared) head; CPU first,
        # Lightning/DDP moves it to the rank's device.
        self.model = build_finetune_model(
            backbone_key, num_labels=1, task_type="regression",
            pooling="mean", device="cpu")
        for p in self.model.parameters():
            p.requires_grad = True
        if grad_ckpt:
            try:
                self.model.gradient_checkpointing_enable()
            except Exception as e:
                print(f"[warn] grad_ckpt enable failed for {backbone_key}: {e}")

        self.val_outputs = []
        self.test_outputs = []

    def forward(self, batch):
        # build_finetune_model's wrapper takes the spec.encode dict + labels
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        out = self.model(**inputs)
        return out["logits"] if isinstance(out, dict) else out

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch).reshape(-1)
        loss = nn.MSELoss()(logits.float(), batch["labels"].float().reshape(-1))
        self.log("train_loss", loss, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # reshape(-1) (not squeeze) so bsz=1 stays 1-D -> torch.cat works in epoch_end
        logits = self.forward(batch).reshape(-1)
        y = batch["labels"].float().reshape(-1)
        loss = nn.MSELoss()(logits.float(), y)
        if dataloader_idx == 0:
            self.log("val_loss", loss, sync_dist=True, add_dataloader_idx=False)
        rec = {"pred": logits.detach().float(), "target": y.detach().float()}
        (self.val_outputs if dataloader_idx == 0 else self.test_outputs).append(rec)
        return loss

    def on_validation_epoch_end(self):
        def process(outs, split):
            if not outs:
                return
            local_p = torch.cat([o["pred"] for o in outs])
            local_t = torch.cat([o["target"] for o in outs])
            all_p = self.all_gather(local_p).view(-1).float().cpu().numpy()
            all_t = self.all_gather(local_t).view(-1).float().cpu().numpy()
            r2 = float(r2_score(all_t, all_p))
            self.log(f"{split}_r2_global", r2, sync_dist=False, prog_bar=(split == "val"))
            if self.trainer.is_global_zero:
                pear = float(pearsonr(all_p, all_t)[0])
                spear = float(spearmanr(all_p, all_t).statistic)
                mse = float(((all_p - all_t) ** 2).mean())
                print(f"\n🌟 {split} ep{self.current_epoch}: R2={r2:.4f} "
                      f"Pearson={pear:.4f} Spearman={spear:.4f} MSE={mse:.4f}", flush=True)
                os.makedirs(self.save_dir, exist_ok=True)
                fn = os.path.join(self.save_dir, f"{split}_ep{self.current_epoch}_r2_{r2:.4f}.csv")
                with open(fn, "w", newline="") as f:
                    w = csv.writer(f); w.writerow(["pred", "target"])
                    for p, t in zip(all_p, all_t):
                        w.writerow([f"{p:.6f}", f"{t:.6f}"])
        process(self.val_outputs, "val")
        process(self.test_outputs, "test")
        self.val_outputs.clear()
        self.test_outputs.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "monitor": "val_r2_global",
                                 "interval": "epoch", "frequency": 1, "strict": False}}
