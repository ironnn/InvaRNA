#!/usr/bin/env python
"""Regenerate Fig. 3g predictions from the frozen model checkpoints.

This is the expensive path. The exact manuscript prediction files are preserved in
``assets/benchmark_data/mpra/fig3g/predictions`` for the fast, byte-verified reproduction path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "assets/benchmark_data/mpra/fig3g/input/mpra_panel_g_3k.parquet"
ASSETS = ROOT / "assets/checkpoints/external/pretrained/models"


def read_input(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path) if path.suffix in {".parquet", ".pq"} else pd.read_csv(path)
    required = {"mrna", "utr5_size", "cds_size", "sample", "ribosome_loading"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    if "utr5_sequence" not in frame:
        frame["utr5_sequence"] = [str(s)[: int(n)] for s, n in zip(frame.mrna, frame.utr5_size)]
        frame["cds_sequence"] = [
            str(s)[int(u): int(u) + int(c)] for s, u, c in zip(frame.mrna, frame.utr5_size, frame.cds_size)
        ]
        frame["utr3_sequence"] = [
            str(s)[int(u) + int(c):] for s, u, c in zip(frame.mrna, frame.utr5_size, frame.cds_size)
        ]
    return frame.reset_index(drop=True)


def output_frame(frame: pd.DataFrame, predictions: list[float], column: str, output: Path) -> None:
    result = frame[["utr5_size", "cds_size", "sample", "ribosome_loading"]].copy()
    result[column] = predictions
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Saved {output} ({len(result)} rows)")


def predict_invarna(frame: pd.DataFrame, device: torch.device, batch_size: int) -> list[float]:
    sys.path.insert(0, str(ROOT / "src"))
    from invarna.evaluation.human_te import load_stage2, run_invarna_inference

    model = load_stage2("final_tdc", device)
    result = run_invarna_inference(
        model, frame, rank=0, world_size=1, device=device,
        batch_size=batch_size, phase="FIG3G_INVARNA",
    )
    return result["_pred"].tolist()


def predict_utrlm(frame: pd.DataFrame, device: torch.device, batch_size: int) -> list[float]:
    code_root = ROOT / "external_models/UTR_LM"
    sys.path.insert(0, str(code_root / "Scripts"))
    sys.path.insert(0, str(code_root))
    from esm.data import Alphabet
    from model_architecture import CNN_linear

    alphabet = Alphabet(standard_toks="AGCT", mask_prob=0.0)
    sequences = frame["utr5_sequence"].astype(str).str.upper().str.replace("U", "T", regex=False)

    def tokenize(sequence: str) -> torch.Tensor:
        sequence = sequence[-100:]
        tokens = [alphabet.cls_idx]
        tokens.extend(alphabet.tok_to_idx.get(base, alphabet.unk_idx) for base in sequence)
        tokens.append(alphabet.eos_idx)
        return torch.tensor(tokens, dtype=torch.long)

    tokenized = [tokenize(sequence) for sequence in sequences]
    sums = np.zeros(len(frame), dtype=np.float64)
    paths = sorted((ASSETS / "UTR_LM/hek_ensemble").glob("hek_fold*.pt"))
    if len(paths) != 10:
        raise FileNotFoundError(f"Expected 10 UTR-LM HEK checkpoints; found {len(paths)}")

    for path in paths:
        model = CNN_linear(
            layers=6, heads=16, embed_dim=128, inp_len=100, nodes=40,
            dropout3=0.2, cnn_layers=0, avg_emb=False, bos_emb=True,
            magic=False, modelfile="ESM2SI_3.1",
        ).to(device)
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict({key.replace("module.", ""): value for key, value in state.items()})
        model.eval()
        fold_predictions = []
        with torch.no_grad():
            for start in tqdm(range(0, len(tokenized), batch_size), desc=path.stem):
                items = tokenized[start:start + batch_size]
                max_length = max(len(item) for item in items)
                batch = torch.stack([
                    torch.cat([
                        item,
                        torch.full((max_length - len(item),), alphabet.padding_idx, dtype=torch.long),
                    ]) if len(item) < max_length else item
                    for item in items
                ]).to(device)
                values = model(batch).squeeze(-1).float().cpu()
                fold_predictions.extend([values.item()] if values.ndim == 0 else values.tolist())
        sums += np.asarray(fold_predictions, dtype=np.float64)
        del model
    return (sums / len(paths)).tolist()


def predict_ribonn(frame: pd.DataFrame, batch_size: int) -> list[float]:
    asset_root = ASSETS / "ribonn"
    sys.path.insert(0, str(asset_root / "RiboNN-1.0.0"))
    sys.path.insert(0, str(asset_root))
    from predict_ribonn_fast import run_ribonn_prediction

    predictions = []
    for start in tqdm(range(0, len(frame), batch_size), desc="RiboNN"):
        batch = frame.iloc[start:start + batch_size].copy().reset_index(drop=True)
        batch["tx_id"] = [f"idx_{index}" for index in range(start, start + len(batch))]
        result = run_ribonn_prediction(batch, n_folds=1)
        by_id = dict(zip(result["tx_id"], result["mean_predicted_TE"]))
        predictions.extend(by_id.get(f"idx_{index}", np.nan) for index in range(start, start + len(batch)))
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("invarna", "ribonn", "utrlm"))
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, help="Diagnostic prefix only; omit for manuscript inference")
    args = parser.parse_args()

    frame = read_input(args.input)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        frame = frame.head(args.limit).copy()
    device = torch.device(args.device)
    if args.model == "invarna":
        predictions = predict_invarna(frame, device, args.batch_size)
        column = "pred_final_tdc"
    elif args.model == "utrlm":
        predictions = predict_utrlm(frame, device, args.batch_size)
        column = "pred_utrlm"
    else:
        if device.type != "cuda":
            raise ValueError("The copied RiboNN implementation requires CUDA")
        predictions = predict_ribonn(frame, args.batch_size)
        column = "pred_ribonn"
    output_frame(frame, predictions, column, args.output)


if __name__ == "__main__":
    main()
