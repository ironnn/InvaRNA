#!/usr/bin/env python
# encoding: utf-8
"""
UTR-LM full-parameter fine-tuning on the pure_te_bench TE dataset (Lightning DDP).

UTR-LM is NOT one of the unified backbones — it's an ESM2-based CNN_linear model
(esm2 6L/128d backbone + CNN + linear regression head, native 1-d output).

Approach (per user): load weights exactly like predict_utrlm.py's TEPredictor
(CNN_linear + HEK fold0 finetuned checkpoint), then UNFREEZE all params and
full-fine-tune on pure_te_bench.

Region : 5'UTR — tokenizer takes the LAST 100 nt of utr5_sequence (INP_LEN=100),
         exactly as predict_utrlm.py does.
MaxLen : 100 nt (fixed by the model).
Env    : mamballm

Harness follows the other train_te_*.py (Lightning + DDP, bf16-mixed):
  - label=mean_te, loss=MSELoss
  - per-epoch val+test: all_gather -> sklearn r2_score -> dump
        {save_dir}/{split}_ep{epoch}_r2_{r2:.4f}.csv  (pred,target)
  - ModelCheckpoint(monitor=val_r2_global, mode=max); metrics.csv via CSVLogger

Run:
  conda run -n mamballm python sft/train_te_utrlm.py \
      --devices 7 --epochs 10 --output_dir ./te_utrlm_out
"""
import os, sys, csv, argparse
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

# UTR-LM lives under models/UTR_LM with its own esm + model_architecture
UTRLM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", "checkpoints", "external", "pretrained", "models", "UTR_LM")
sys.path.insert(0, os.path.join(UTRLM_DIR, "Scripts"))
sys.path.insert(0, UTRLM_DIR)
from esm.data import Alphabet
from model_architecture import CNN_linear

# CNN_linear hyperparams — identical to predict_utrlm.py / MJ4 training
LAYERS, HEADS, EMBED_DIM, INP_LEN, NODES = 6, 16, 128, 100, 40
DROPOUT3, CNN_LAYERS, AVG_EMB, BOS_EMB, MAGIC, MODELFILE = 0.2, 0, False, True, False, "ESM2SI_3.1"
FOLD0_CKPT = os.path.join(
    UTRLM_DIR, "Model/Downstream/TE_EL/"
    "MJ4_seed1337_TE_ESM2SI_3.1.1e-2.H.dropout2_HEK_te_log_utr_seqlen100_"
    "AvgEmbFalse_BosEmbTrue_CNNlayer0_epoch300_patiences0_nodes40_dropout30.2_"
    "finetuneTrue_huberlossTrue_magicFalse_lr0.01_fold0_epoch275.pt")

_ALPHA = Alphabet(standard_toks="AGCT", mask_prob=0.0)


def tokenize(seq):
    """5'UTR last-100nt tokenization, exactly as predict_utrlm.py."""
    seq = seq.upper().replace("U", "T")[-INP_LEN:]
    toks = [_ALPHA.cls_idx]
    toks += [_ALPHA.tok_to_idx.get(c, _ALPHA.unk_idx) for c in seq]
    toks.append(_ALPHA.eos_idx)
    return torch.tensor(toks, dtype=torch.long)


def load_split(path):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["mrna", "utr5_size", "mean_te"]).to_pandas()
    seqs, labs = [], []
    for _, r in t.iterrows():
        utr5 = r["mrna"][: int(r["utr5_size"])]   # 5'UTR region
        if len(utr5) == 0:
            continue
        seqs.append(utr5); labs.append(float(r["mean_te"]))
    return seqs, np.asarray(labs, dtype=np.float64)


class UTRDS(Dataset):
    def __init__(self, seqs, labs):
        self.toks = [tokenize(s) for s in seqs]
        self.labs = labs
    def __len__(self): return len(self.toks)
    def __getitem__(self, i):
        return {"input_ids": self.toks[i], "labels": torch.tensor(self.labs[i], dtype=torch.float32)}


def collate(batch):
    toks = [b["input_ids"] for b in batch]
    maxlen = max(len(t) for t in toks)
    padded = torch.full((len(toks), maxlen), _ALPHA.padding_idx, dtype=torch.long)
    for i, t in enumerate(toks):
        padded[i, :len(t)] = t
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_ids": padded, "labels": labels}


class UTRLMLightning(pl.LightningModule):
    def __init__(self, save_dir, lr=1e-5, weight_decay=0.01):
        super().__init__()
        self.save_dir = save_dir; self.lr = lr; self.weight_decay = weight_decay
        self.model = CNN_linear(
            layers=LAYERS, heads=HEADS, embed_dim=EMBED_DIM, inp_len=INP_LEN,
            nodes=NODES, dropout3=DROPOUT3, cnn_layers=CNN_LAYERS,
            avg_emb=AVG_EMB, bos_emb=BOS_EMB, magic=MAGIC, modelfile=MODELFILE)
        # load HEK fold0 finetuned weights (like TEPredictor), then unfreeze all
        sd = torch.load(FOLD0_CKPT, map_location="cpu")
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        self.model.load_state_dict(sd)
        for p in self.model.parameters():
            p.requires_grad = True
        self.val_outputs = []; self.test_outputs = []

    def forward(self, input_ids):
        return self.model(input_ids)

    def _step_logits(self, batch):
        return self.forward(batch["input_ids"]).reshape(-1)

    def training_step(self, batch, _):
        loss = nn.MSELoss()(self._step_logits(batch).float(), batch["labels"].float().reshape(-1))
        self.log("train_loss", loss, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self._step_logits(batch); y = batch["labels"].float().reshape(-1)
        loss = nn.MSELoss()(logits.float(), y)
        if dataloader_idx == 0:
            self.log("val_loss", loss, sync_dist=True, add_dataloader_idx=False)
        rec = {"pred": logits.detach().float(), "target": y.detach().float()}
        (self.val_outputs if dataloader_idx == 0 else self.test_outputs).append(rec)
        return loss

    def on_validation_epoch_end(self):
        def process(outs, split):
            if not outs: return
            lp = torch.cat([o["pred"] for o in outs]); lt = torch.cat([o["target"] for o in outs])
            ap = self.all_gather(lp).reshape(-1).float().cpu().numpy()
            at = self.all_gather(lt).reshape(-1).float().cpu().numpy()
            r2 = float(r2_score(at, ap))
            self.log(f"{split}_r2_global", r2, sync_dist=False, prog_bar=(split == "val"))
            if self.trainer.is_global_zero:
                pear = float(pearsonr(ap, at)[0]); spear = float(spearmanr(ap, at).statistic)
                mse = float(((ap - at) ** 2).mean())
                print(f"\n🌟 {split} ep{self.current_epoch}: R2={r2:.4f} "
                      f"Pearson={pear:.4f} Spearman={spear:.4f} MSE={mse:.4f}", flush=True)
                os.makedirs(self.save_dir, exist_ok=True)
                fn = os.path.join(self.save_dir, f"{split}_ep{self.current_epoch}_r2_{r2:.4f}.csv")
                with open(fn, "w", newline="") as f:
                    w = csv.writer(f); w.writerow(["pred", "target"])
                    for p, t in zip(ap, at): w.writerow([f"{p:.6f}", f"{t:.6f}"])
        process(self.val_outputs, "val"); process(self.test_outputs, "test")
        self.val_outputs.clear(); self.test_outputs.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad),
                                lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_r2_global",
                "interval": "epoch", "frequency": 1, "strict": False}}


class UTRDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, bsz, num_workers=4, max_train=-1):
        super().__init__()
        self.data_dir = data_dir; self.bsz = bsz; self.num_workers = num_workers; self.max_train = max_train
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
        self.tr = UTRDS(tr_s, tr_y); self.val = UTRDS(val_s, val_y); self.test = UTRDS(test_s, test_y)
        print(f"[data] train={len(self.tr)} val={len(self.val)} test={len(self.test)}", flush=True)
    def train_dataloader(self):
        return DataLoader(self.tr, batch_size=self.bsz, shuffle=True, collate_fn=collate, num_workers=self.num_workers)
    def val_dataloader(self):
        return [DataLoader(self.val, batch_size=self.bsz, shuffle=False, collate_fn=collate, num_workers=self.num_workers),
                DataLoader(self.test, batch_size=self.bsz, shuffle=False, collate_fn=collate, num_workers=self.num_workers)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="assets/training_data/common_te_splits")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bsz", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--devices", type=int, default=7)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--output_dir", default="./te_utrlm_out")
    ap.add_argument("--seed", type=int, default=2222)
    ap.add_argument("--max_train", type=int, default=-1)
    args = ap.parse_args()

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[TE-utrlm] region=5'UTR(last{INP_LEN}nt) env=mamballm devices={args.devices}", flush=True)

    dm = UTRDataModule(args.data_dir, args.bsz, args.num_workers, args.max_train)
    model = UTRLMLightning(save_dir=args.output_dir, lr=args.lr, weight_decay=args.weight_decay)
    n_tr = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in model.model.parameters())
    print(f"[full-FT] trainable {n_tr:,}/{n_tot:,} ({100*n_tr/n_tot:.1f}%)", flush=True)

    ckpt = ModelCheckpoint(dirpath=args.output_dir, filename="best-{epoch:02d}-{val_r2_global:.4f}",
                           monitor="val_r2_global", mode="max", save_top_k=1, save_weights_only=True)
    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator="gpu", devices=args.devices,
        # CNN_linear's ESM2 has aux heads (contact/structure) not used in regression
        # forward -> DDP needs find_unused_parameters=True.
        strategy=DDPStrategy(find_unused_parameters=True) if args.devices > 1 else "auto",
        precision="bf16-mixed", log_every_n_steps=10,
        callbacks=[ckpt], logger=CSVLogger(save_dir=args.output_dir, name="", version=""),
        accumulate_grad_batches=args.grad_accum, use_distributed_sampler=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=dm)
    if trainer.is_global_zero:
        print(f"BEST val_r2_global = {ckpt.best_model_score} @ {ckpt.best_model_path}", flush=True)


if __name__ == "__main__":
    main()
