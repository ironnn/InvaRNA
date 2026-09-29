"""
pretrain_module.py — self-contained MLM LightningModule for InvaRNA backbone pretraining.

Replaces the 390-line state-spaces SequenceLightningModule + LMTask. Modeled on
the validated SFT module, but for masked language modeling.

Faithful to the original training recipe:
  - loss = F.cross_entropy(logits.view(-1,C), target.view(-1), ignore_index=PAD=4)
    -> loss only on masked positions (target=pad elsewhere, set by mlm_getitem)
  - AdamW lr=0.008, weight_decay=0.1, betas=(0.9,0.95); bias & normalization params
    get weight_decay=0.0 (matches add_optimizer_hooks)
  - scheduler = TimmCosineLRScheduler (t_initial=9000, lr_min=1e-4, warmup_t=1000,
    warmup_lr_init=1e-6, warmup_prefix=True), stepped PER STEP (t_in_epochs=False)
  - logs `trainer/loss` (per-step, monitor for the every-250-step checkpoint) and
    `val/loss` (epoch, monitor for best checkpoint)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .schedulers import TimmCosineLRScheduler


class MLMPretrainModule(pl.LightningModule):
    def __init__(self, model: nn.Module, pad_token_id: int = 4,
                 lr: float = 0.008, weight_decay: float = 0.1,
                 betas=(0.9, 0.95),
                 t_initial: int = 9000, lr_min: float = 1e-4,
                 warmup_t: int = 1000, warmup_lr_init: float = 1e-6,
                 warmup_prefix: bool = True):
        super().__init__()
        self.model = model
        self.pad_token_id = pad_token_id
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = tuple(betas)
        self.sched_kwargs = dict(
            t_initial=t_initial, lr_min=lr_min,
            warmup_t=warmup_t, warmup_lr_init=warmup_lr_init,
            warmup_prefix=warmup_prefix, t_in_epochs=False,
        )
        self.save_hyperparameters(ignore=["model"])

    def forward(self, input_ids):
        return self.model(input_ids=input_ids).logits  # (B, L, vocab)

    def _mlm_loss(self, logits, target):
        # only masked positions contribute (target=pad_token_id elsewhere)
        return F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            target.view(-1),
            ignore_index=self.pad_token_id,
        )

    def training_step(self, batch, batch_idx):
        data, target = batch
        logits = self(data)
        loss = self._mlm_loss(logits, target)
        # `trainer/loss` is the monitor for the every-250-step checkpoint
        self.log("trainer/loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        logits = self(data)
        loss = self._mlm_loss(logits, target)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # also expose test/loss alias (pretrain val/test share the same split)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    # --- optimizer: AdamW with bias/normalization params at weight_decay=0 ---
    def configure_optimizers(self):
        decay, no_decay = [], []
        # mark params on bias / normalization / embedding modules as no-weight-decay
        norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm,
                      nn.SyncBatchNorm, nn.InstanceNorm1d, nn.InstanceNorm2d,
                      nn.InstanceNorm3d, nn.LayerNorm, nn.LocalResponseNorm, nn.Embedding)
        seen = set()
        for mn, m in self.model.named_modules():
            is_norm = isinstance(m, norm_types)
            for pn, p in m.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                if pn.endswith("bias") or is_norm or getattr(p, "_no_weight_decay", False):
                    no_decay.append(p)
                else:
                    decay.append(p)
        param_groups = [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=self.lr, betas=self.betas)
        scheduler = TimmCosineLRScheduler(optimizer, **self.sched_kwargs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # timm scheduler isn't a torch native scheduler -> step it by global_step
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.global_step)
