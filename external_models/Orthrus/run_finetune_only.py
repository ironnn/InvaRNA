import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "orthrus"))

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from run_mrl_eval import (
    load_mrl_data, homology_split, load_orthrus_model,
    MRLDataset, MRLFinetuneModel, SEED, ASSETS_DIR
)

pl.seed_everything(SEED)
torch.set_float32_matmul_precision("medium")

X, y, genes = load_mrl_data()
train_X, train_y, val_X, val_y, test_X, test_y = homology_split(X, y, genes)

train_ds = MRLDataset(train_X, train_y)
val_ds = MRLDataset(val_X, val_y)
test_ds = MRLDataset(test_X, test_y)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

pretrained = load_orthrus_model(gradient_checkpointing=True)
model = MRLFinetuneModel(pretrained, lr=1e-4, warmup_steps=500, total_steps=6000)

checkpoint_cb = pl.callbacks.ModelCheckpoint(
    monitor="val_loss", mode="min", save_top_k=1, filename="best-{epoch}-{step}"
)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    precision="bf16-mixed",
    max_steps=6000,
    callbacks=[checkpoint_cb],
    gradient_clip_val=10.0,
    gradient_clip_algorithm="norm",
    logger=pl.loggers.CSVLogger(save_dir=os.path.join(ASSETS_DIR, "..", "runs"), name="mrl_finetune"),
    enable_progress_bar=True,
)

trainer.fit(model, train_loader, val_loader)

print("\n--- Test Set Evaluation (Best Checkpoint) ---")
best_model = MRLFinetuneModel.load_from_checkpoint(
    checkpoint_cb.best_model_path,
    pretrained_model=load_orthrus_model(gradient_checkpointing=True),
    lr=1e-3, warmup_steps=1000, total_steps=6000,
)
results = trainer.test(best_model, test_loader)
print(results)
