#!/usr/bin/env python
"""Run a selected external-model TE checkpoint on the held-out human test set.

Run this from the repository root in the model-specific environment.  The code
loads the actual copied training module so region extraction and tokenization are
identical to the completed run.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "external_models"
sys.path.insert(0, str(ROOT))

COMMON_MODELS = {
    "codonbert": ("train_te_codonbert.py", 1024),
    "dnabert2": ("train_te_dnabert2.py", 1024),
    "lucaone": ("train_te_lucaone.py", 1280),
    "mrnabert": ("train_te_mrnabert.py", 1024),
    "orthrus": ("train_te_orthrus.py", 1024),
    "rnafm": ("train_te_rnafm.py", 1024),
}
ALL_MODELS = list(COMMON_MODELS) + ["evo2_8k", "ribonn", "utrlm"]


def load_training_module(filename: str):
    path = BASELINES / "training" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predict_common(model_key: str, checkpoint: Path, test_path: Path, batch_size: int):
    import torch
    from torch.utils.data import DataLoader

    from external_models.common.backbones import get_spec
    from external_models.common.te_lightning import TELightningModule

    filename, max_len = COMMON_MODELS[model_key]
    training = load_training_module(filename)
    backbone = get_spec(model_key)
    tokenizer = backbone.load_tokenizer()
    sequences, targets = training.load_split(str(test_path))
    dataset = training.SeqDS(sequences, targets, backbone, tokenizer, max_len)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0 if tokenizer is not None else 0
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=training.make_collator(pad_id),
        num_workers=0,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TELightningModule.load_from_checkpoint(str(checkpoint), map_location="cpu")
    model = model.to(device).eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            predictions.extend(model.forward(batch).reshape(-1).float().cpu().tolist())
    return predictions, targets.tolist()


def predict_evo2(model_key: str, checkpoint: Path, test_path: Path):
    import torch

    training = load_training_module("train_te_evo2.py")
    from external_models.common.evo2_ft import Evo2ForSeqCls

    sequences, targets = training.load_split(str(test_path), 8000)
    model = Evo2ForSeqCls(num_labels=1, task_type="regression", pooling="mean")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
    predictions = training.predict(model, sequences, max_len)
    return predictions.tolist(), targets.tolist()


def predict_ribonn(checkpoint: Path, test_path: Path, batch_size: int):
    import torch

    training = load_training_module("train_te_ribonn.py")
    import pandas as pd

    runs = pd.read_csv(Path(training.RIBONN_ROOT) / "models/runs.csv").head(1)
    run_id = runs.run_id.iloc[0]
    config = training.extract_config(runs, run_id)
    config["max_utr5_len"] = training.MAX_UTR5_LEN
    config["max_cds_utr3_len"] = training.MAX_CDS_UTR3_LEN
    model = training.RiboNN(**config)
    model.head[-1] = torch.nn.Linear(model.head[-1].in_features, 1)
    model.num_targets = 1
    model.loss = torch.nn.functional.mse_loss
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    frame = training.load_split(str(test_path))
    dataset = training.make_dataset(frame, config)
    predictions = training.predict(model, dataset, batch_size, device)
    return predictions.tolist(), frame.mean_te.astype(float).tolist()


def predict_utrlm(checkpoint: Path, test_path: Path, batch_size: int):
    import torch
    from torch.utils.data import DataLoader

    training = load_training_module("train_te_utrlm.py")
    sequences, targets = training.load_split(str(test_path))
    dataset = training.UTRDS(sequences, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=training.collate, num_workers=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = training.UTRLMLightning.load_from_checkpoint(
        str(checkpoint), map_location="cpu", save_dir=".")
    model = model.to(device).eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            predictions.extend(
                model(batch["input_ids"].to(device)).reshape(-1).float().cpu().tolist()
            )
    return predictions, targets.tolist()


def r2_score(targets: list[float], predictions: list[float]) -> float:
    mean_target = sum(targets) / len(targets)
    residual = sum((truth - estimate) ** 2 for truth, estimate in zip(targets, predictions))
    total = sum((truth - mean_target) ** 2 for truth in targets)
    return 1.0 - residual / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=ALL_MODELS)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--test-data", type=Path,
        default=ROOT / "assets/training_data/common_te_splits/human_test_wt.parquet",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    default_checkpoint = ROOT / "assets/checkpoints/te_finetuned" / args.model / (
        "best.pt" if args.model.startswith("evo2") or args.model == "ribonn" else "best.ckpt"
    )
    checkpoint = args.checkpoint or default_checkpoint
    if args.model in COMMON_MODELS:
        predictions, targets = predict_common(args.model, checkpoint, args.test_data, args.batch_size)
    elif args.model.startswith("evo2"):
        predictions, targets = predict_evo2(args.model, checkpoint, args.test_data)
    elif args.model == "ribonn":
        predictions, targets = predict_ribonn(checkpoint, args.test_data, args.batch_size)
    else:
        predictions, targets = predict_utrlm(checkpoint, args.test_data, args.batch_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pred", "target"])
        writer.writerows(zip(predictions, targets))
    print(f"model={args.model} n={len(predictions)} test_R2={r2_score(targets, predictions):.6f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
