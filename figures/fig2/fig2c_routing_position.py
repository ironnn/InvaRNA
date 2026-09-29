#!/usr/bin/env python
"""Recompute Fig. 2C Motif-MoE routing from the frozen backbone and exact inputs."""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from invarna.models.configuration import InvaRNAConfig
from invarna.models.mamba_backbone import InvaRNAForMaskedLM


EXPERTS = ["Single-base", "Codon", "9-mer", "Ribosome"]


def load_masked_inputs(path: Path) -> tuple[torch.Tensor, list[str]]:
    rows = []
    record_ids = []
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            record_ids.append(row["fasta_record_id"])
            rows.append([int(value) for value in row["input_ids"].split()])
    inputs = torch.tensor(rows, dtype=torch.long)
    if inputs.shape != (24, 10000):
        raise RuntimeError(f"expected a 24 x 10000 input fixture, found {tuple(inputs.shape)}")
    return inputs, record_ids


def build_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    config = OmegaConf.load(config_path)
    model_config = OmegaConf.to_container(config.model.config, resolve=True)
    model_config.pop("_target_", None)
    model = InvaRNAForMaskedLM(InvaRNAConfig(**model_config))
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    return model.to(device).eval()


def recompute(
    input_ids: torch.Tensor,
    model: InvaRNAForMaskedLM,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_ids = input_ids.to(device)
    with torch.inference_mode():
        _hidden, _states, moe_logits = model.invaRNA(
            input_ids=input_ids, output_hidden_states=True, return_dict=False
        )
    probabilities = torch.softmax(moe_logits.float(), dim=-1)
    valid = input_ids.ne(4)
    masked_probabilities = probabilities * valid.unsqueeze(-1)
    position_counts = valid.float().sum(dim=0)
    safe_counts = position_counts.clamp(min=1)
    mean_curve = (masked_probabilities.sum(dim=0) / safe_counts.unsqueeze(-1)).cpu().numpy()

    curve = pd.DataFrame({"sequence_position_bp": np.arange(8000, dtype=int)})
    curve["region"] = np.where(
        curve["sequence_position_bp"].lt(1000), "5'UTR", "CDS + 3'UTR"
    )
    curve["non_padding_sequence_count"] = position_counts[:8000].cpu().numpy().astype(int)
    for expert_index, expert in enumerate(EXPERTS):
        raw = mean_curve[:8000, expert_index]
        curve[f"{expert}_raw_mean_probability"] = raw
        curve[f"{expert}_rolling80_mean_probability"] = (
            pd.Series(raw).rolling(window=80, center=True, min_periods=1).mean().to_numpy()
        )

    bar_rows = []
    for region, start, end in (("5'UTR", 0, 1000), ("CDS + 3'UTR", 1000, 9000)):
        region_valid = valid[:, start:end]
        denominator = int(region_valid.sum().item())
        sums = (
            probabilities[:, start:end, :] * region_valid.unsqueeze(-1)
        ).sum(dim=(0, 1)).cpu().numpy()
        for expert_index, expert in enumerate(EXPERTS):
            bar_rows.append(
                {
                    "region": region,
                    "expert": expert,
                    "expert_order": expert_index + 1,
                    "probability_sum": float(sums[expert_index]),
                    "valid_token_count": denominator,
                    "mean_expert_probability": float(sums[expert_index] / denominator),
                }
            )
    return curve, pd.DataFrame(bar_rows)


def compare_to_archive(
    curve: pd.DataFrame,
    bars: pd.DataFrame,
    source_dir: Path,
    curve_tolerance: float,
    bar_tolerance: float,
) -> dict[str, float]:
    expected_curve = pd.read_csv(source_dir / "F2C_left_routing_curve.csv")
    expected_bars = pd.read_csv(source_dir / "F2C_right_region_specialization.csv")
    if not curve[["sequence_position_bp", "region", "non_padding_sequence_count"]].equals(
        expected_curve[["sequence_position_bp", "region", "non_padding_sequence_count"]]
    ):
        raise RuntimeError("Fig. 2C position/region/mask counts differ from the archive")
    if not bars[["region", "expert", "expert_order", "valid_token_count"]].equals(
        expected_bars[["region", "expert", "expert_order", "valid_token_count"]]
    ):
        raise RuntimeError("Fig. 2C region/expert/token counts differ from the archive")

    curve_columns = [column for column in curve if column.endswith("_probability")]
    curve_delta = float(
        np.max(np.abs(curve[curve_columns].to_numpy() - expected_curve[curve_columns].to_numpy()))
    )
    bar_delta = float(
        np.max(
            np.abs(
                bars["mean_expert_probability"].to_numpy()
                - expected_bars["mean_expert_probability"].to_numpy()
            )
        )
    )
    if curve_delta > curve_tolerance or bar_delta > bar_tolerance:
        raise RuntimeError(
            "Fig. 2C replay exceeded tolerance: "
            f"curve={curve_delta:.6g} (limit {curve_tolerance}), "
            f"bar={bar_delta:.6g} (limit {bar_tolerance})"
        )
    return {"curve_max_abs_delta": curve_delta, "bar_max_abs_delta": bar_delta}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = ROOT / "assets/manuscript_figures/fig2_compact"
    parser.add_argument(
        "--inputs", type=Path, default=source / "F2C_seed42_masked_input_ids.tsv.gz"
    )
    parser.add_argument(
        "--manifest", type=Path, default=source / "F2C_sample_manifest_seed42.csv"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "backbone/configs/mamba_motif_moe.yaml"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "assets/checkpoints/backbone/pretrained_step13500.pt",
    )
    parser.add_argument("--source-dir", type=Path, default=source)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/fig2/f2c_replay")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--curve-tolerance", type=float, default=5e-4)
    parser.add_argument("--bar-tolerance", type=float, default=5e-5)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the full Fig. 2C replay; use saved plot data for CPU-only reproduction")
    input_ids, record_ids = load_masked_inputs(args.inputs)
    manifest = pd.read_csv(args.manifest).sort_values("batch_order")
    if record_ids != manifest["fasta_record_id"].tolist():
        raise RuntimeError("masked-input record order differs from the archived manifest")

    model = build_model(args.config, args.checkpoint, device)
    curve, bars = recompute(input_ids, model, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    deltas = compare_to_archive(
        curve, bars, args.source_dir, args.curve_tolerance, args.bar_tolerance
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(args.output_dir / "F2C_left_routing_curve.csv", index=False, float_format="%.10g")
    bars.to_csv(
        args.output_dir / "F2C_right_region_specialization.csv",
        index=False,
        float_format="%.10g",
    )
    print(
        "PASS: frozen-backbone Fig. 2C replay; "
        f"curve max |delta|={deltas['curve_max_abs_delta']:.6g}, "
        f"bar max |delta|={deltas['bar_max_abs_delta']:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
