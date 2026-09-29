"""
MRL evaluation script: Homology-split linear probe + end-to-end finetune.
Uses Orthrus 6-track large model (pytorch_model.bin).
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orthrus.mamba import MixerModel, mean_unpadded
from orthrus.layers import ProjectionHead
from orthrus.util import train_test_split_homologous, load_homology_df
from orthrus.eval_utils import get_representations, get_unpadded_seq_lens, PearsonR, SpearmanR

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
MODEL_PATH = os.path.join(ASSETS_DIR, "pytorch_model.bin")
DATA_PATH = os.path.join(ASSETS_DIR, "mrl_isoform_resolved.npz")

SEED = 2547
D_MODEL = 512
N_LAYER = 6
N_TRACKS = 6


def load_orthrus_model(gradient_checkpointing=False):
    model = MixerModel(d_model=D_MODEL, n_layer=N_LAYER, input_dim=N_TRACKS,
                       gradient_checkpointing=gradient_checkpointing)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def load_mrl_data():
    data = np.load(DATA_PATH, allow_pickle=True)
    X = data["X"].astype(np.float32)  # (N, 12288, 6) channel_last
    y = data["y"].astype(np.float32)
    genes = data["genes"]
    return X, y, genes


def homology_split(X, y, genes, seed=SEED):
    homology_df = load_homology_df("human")
    np.random.seed(seed)

    split1 = train_test_split_homologous(genes, homology_df, test_size=0.30, random_state=seed)
    train_X = X[split1["train_indices"]]
    train_y = y[split1["train_indices"]]
    vt_X = X[split1["test_indices"]]
    vt_y = y[split1["test_indices"]]
    vt_genes = genes[split1["test_indices"]]

    split2 = train_test_split_homologous(vt_genes, homology_df, test_size=0.50, random_state=seed)
    val_X = vt_X[split2["train_indices"]]
    val_y = vt_y[split2["train_indices"]]
    test_X = vt_X[split2["test_indices"]]
    test_y = vt_y[split2["test_indices"]]

    print(f"Homology split: Train={train_X.shape[0]}, Val={val_X.shape[0]}, Test={test_X.shape[0]}")
    return train_X, train_y, val_X, val_y, test_X, test_y


# ==================== Part 1: Linear Probe ====================

def run_linear_probe():
    print("\n" + "=" * 60)
    print("Part 1: Homology-Split Linear Probe")
    print("=" * 60)

    model = load_orthrus_model()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X, y, genes = load_mrl_data()
    train_X, train_y, val_X, val_y, test_X, test_y = homology_split(X, y, genes)

    print("Extracting representations...")
    train_emb = get_representations(model, train_X, batch_size=16, channel_last=True)
    val_emb = get_representations(model, val_X, batch_size=16, channel_last=True)
    test_emb = get_representations(model, test_X, batch_size=16, channel_last=True)

    print(f"Embeddings: train={train_emb.shape}, val={val_emb.shape}, test={test_emb.shape}")

    print("Fitting RidgeCV...")
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    ridge = RidgeCV(alphas=alphas).fit(train_emb, train_y)
    print(f"Best alpha: {ridge.alpha_}")

    print("\n--- Linear Probe Results (Homology Split) ---")
    for name, emb, true in [("Train", train_emb, train_y), ("Val", val_emb, val_y), ("Test", test_emb, test_y)]:
        pred = ridge.predict(emb)
        mse = np.mean((true - pred) ** 2)
        r = pearsonr(pred, true).statistic
        rho = spearmanr(pred, true).statistic
        print(f"  {name}: MSE={mse:.4f}, PearsonR={r:.4f}, SpearmanR={rho:.4f}")


# ==================== Part 2: Finetune ====================

class MRLDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, L, C) channel_last -> transpose to (N, C, L) for model
        self.X = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        summed = (np.sum(X, axis=-1) >= 1).astype(int)
        reverse_sums = np.flip(summed, axis=-1)
        first_occurs = np.argmax(reverse_sums, axis=1)
        self.lengths = torch.tensor(X.shape[1] - first_occurs, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.lengths[idx]


class MRLFinetuneModel(pl.LightningModule):
    def __init__(self, pretrained_model, lr=1e-3, warmup_steps=1000, total_steps=6000):
        super().__init__()
        self.model = pretrained_model
        self.projection_head = ProjectionHead(
            input_features=D_MODEL,
            projection_body=256,
            projection_head_size=1,
            norm_type="batchnorm",
            n_layers=2,
            output_bias=True,
            output_sigmoid=False,
        )
        self.loss_fn = nn.MSELoss()
        self.pearsonr = PearsonR()
        self.spearmanr = SpearmanR()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def forward(self, x, lengths):
        rep = self.model.representation(x, lengths)
        return self.projection_head(rep)

    def training_step(self, batch, batch_idx):
        x, y, lengths = batch
        pred = self(x, lengths)
        loss = self.loss_fn(pred, y)
        self.pearsonr.update(pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        return loss

    def on_train_epoch_end(self):
        self.log("train_pearsonr", self.pearsonr.compute(), prog_bar=True)
        self.pearsonr.reset()
        self.spearmanr.reset()

    def validation_step(self, batch, batch_idx):
        x, y, lengths = batch
        pred = self(x, lengths)
        loss = self.loss_fn(pred, y)
        self.pearsonr.update(pred, y)
        self.spearmanr.update(pred, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=x.shape[0])

    def on_validation_epoch_end(self):
        self.log("val_pearsonr", self.pearsonr.compute(), prog_bar=True)
        self.log("val_spearmanr", self.spearmanr.compute(), prog_bar=True)
        self.pearsonr.reset()
        self.spearmanr.reset()

    def test_step(self, batch, batch_idx):
        x, y, lengths = batch
        pred = self(x, lengths)
        loss = self.loss_fn(pred, y)
        self.pearsonr.update(pred, y)
        self.spearmanr.update(pred, y)
        self.log("test_loss", loss, on_epoch=True, batch_size=x.shape[0])
        self.log("test_pearsonr", self.pearsonr.compute(), on_epoch=True)
        self.log("test_spearmanr", self.spearmanr.compute(), on_epoch=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, total_iters=self.warmup_steps),
                CosineAnnealingLR(optimizer, T_max=self.total_steps - self.warmup_steps),
            ],
            milestones=[self.warmup_steps],
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}


def run_finetune():
    print("\n" + "=" * 60)
    print("Part 2: End-to-End Finetune (Homology Split)")
    print("=" * 60)

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
    model = MRLFinetuneModel(pretrained, lr=1e-3, warmup_steps=1000, total_steps=6000)

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1, filename="best-{epoch}-{step}"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=7,
        strategy="ddp",
        precision="16-mixed",
        max_steps=6000,
        callbacks=[checkpoint_cb],
        gradient_clip_val=10.0,
        gradient_clip_algorithm="norm",
        logger=pl.loggers.CSVLogger(save_dir=os.path.join(ASSETS_DIR, "..", "runs"), name="mrl_finetune"),
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader)

    # Evaluate best model on test set
    print("\n--- Test Set Evaluation (Best Checkpoint) ---")
    best_model = MRLFinetuneModel.load_from_checkpoint(
        checkpoint_cb.best_model_path,
        pretrained_model=load_orthrus_model(gradient_checkpointing=True),
        lr=1e-3, warmup_steps=1000, total_steps=6000,
    )
    results = trainer.test(best_model, test_loader)
    print(results)


if __name__ == "__main__":
    run_linear_probe()
    run_finetune()
