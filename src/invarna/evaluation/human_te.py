"""InvaRNA Stage1/Stage2 translation-efficiency inference.

Single GPU:
    python inference/predict_te.py --model w0 --input data.parquet --output pred.csv

Multi GPU:
    torchrun --standalone --nproc_per_node=4 inference/predict_te.py \
        --model w0 --input data.parquet --output pred.csv

Input CSV/parquet files require ``mrna`` and ``utr5_size``. Output CSV files omit the
large ``mrna`` column and add ``pred_<model>`` unless ``--pred_col`` is supplied.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_ROOT = Path(
    os.environ.get("INVARNA_CHECKPOINT_ROOT", ROOT / "assets/checkpoints")
).expanduser().resolve()

sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm

from invarna.models.configuration import InvaRNAConfig
from invarna.models.mamba_backbone import InvaRNAForMaskedLM, InvaRNAForRegression
from invarna.models.tokenization import InvaRNATokenizer
from invarna.models.regression_head import MambaRCHeadStage2
from invarna.data.fixed_frame import TOTAL_LENGTH, align_and_pad

# Released checkpoints contain pickle metadata with the pre-publication module names.
# These aliases point only to the native modules above; no legacy implementation is vendored.
from invarna import models as _models_package
from invarna.models import configuration as _configuration_module
from invarna.models import mamba_backbone as _backbone_module
from invarna.models import tokenization as _tokenization_module

sys.modules.setdefault("backbone", _models_package)
sys.modules.setdefault("backbone.configuration", _configuration_module)
sys.modules.setdefault("backbone.modeling", _backbone_module)
sys.modules.setdefault("backbone.tokenization", _tokenization_module)

BACKBONE_CONFIG = ROOT / "backbone" / "configs" / "mamba_motif_moe.yaml"
BACKBONE_WEIGHTS = CHECKPOINT_ROOT / "backbone" / "pretrained_step13500.pt"

STAGE2_CKPTS = {
    "w2": CHECKPOINT_ROOT / "te_student/w2.ckpt",
    "world_old": CHECKPOINT_ROOT / "provenance/legacy_registry/te_student/world_old.ckpt",
    "w1": CHECKPOINT_ROOT / "te_student/w1.ckpt",
    "noevo": CHECKPOINT_ROOT / "provenance/legacy_registry/te_student/noevo.ckpt",
    "hl_new": CHECKPOINT_ROOT / "half_life/final_half_life.ckpt",
    "old1": CHECKPOINT_ROOT / "provenance/legacy_registry/te_student/old1.ckpt",
    "old2": CHECKPOINT_ROOT / "provenance/legacy_registry/te_student/old2.ckpt",
    "w0": CHECKPOINT_ROOT / "te_student/w0.ckpt",
    "notaylor": CHECKPOINT_ROOT / "provenance/legacy_registry/te_student/notaylor.ckpt",
    "wt_human": CHECKPOINT_ROOT / "provenance/legacy_registry/te_student/wt_human.ckpt",
    "taylor_dist": CHECKPOINT_ROOT / "provenance/legacy_registry/te_student/taylor_dist.ckpt",
    "taylor_dist_e14": CHECKPOINT_ROOT / "provenance/legacy_registry/te_student/taylor_dist_e14.ckpt",
    "only_taylor_beidian": CHECKPOINT_ROOT / "provenance/legacy_registry/te_student/only_taylor_beidian.ckpt",
    "taylor_dist_e21": CHECKPOINT_ROOT / "provenance/legacy_registry/te_student/taylor_dist_e21.ckpt",
    "final_tdc": CHECKPOINT_ROOT / "te_student" / "final_tdc.ckpt",
    "hl0704": CHECKPOINT_ROOT / "provenance/legacy_registry/half_life/hl0704.ckpt",
}

STAGE1_CKPTS = {
    "evo": CHECKPOINT_ROOT / "ablations/stage1/evo.ckpt",
    "noevo_s1": CHECKPOINT_ROOT / "ablations/stage1/noevo_s1.ckpt",
    "test_s1": CHECKPOINT_ROOT / "ablations/stage1/test_s1.ckpt",
    "hlevo": CHECKPOINT_ROOT / "ablations/stage1/hlevo_e2.ckpt",
    "hlevo_e7": CHECKPOINT_ROOT / "ablations/stage1/hlevo_e7.ckpt",
}

ALL_MODELS = list(STAGE2_CKPTS) + list(STAGE1_CKPTS)
ALL_MODELS_WITH_CUSTOM = ALL_MODELS + ["stage2_ckpt"]


def _model_config():
    config = OmegaConf.to_container(
        OmegaConf.load(BACKBONE_CONFIG).model.config, resolve=True
    )
    config.pop("_target_", None)
    return InvaRNAConfig(**config)


def setup_dist():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
        torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


def _checkpoint_state(path: Path, device):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(str(path), map_location=device, weights_only=False)
    return payload.get("state_dict", payload)


def load_stage2(model_key: str, device, ckpt_override: str = None):
    ckpt_path = Path(ckpt_override) if ckpt_override else STAGE2_CKPTS[model_key]
    config = _model_config()
    backbone = InvaRNAForMaskedLM(config).invaRNA
    model = MambaRCHeadStage2(pretrained_backbone=backbone, config=config)
    model.load_state_dict(_checkpoint_state(ckpt_path, device), strict=True)
    model.to(device).eval()
    print(f"[stage2] Loaded [{model_key}] from {ckpt_path}", flush=True)
    return model


class _Stage1Model(nn.Module):
    """Checkpoint-compatible wrapper around the single-stage regression model."""

    def __init__(self):
        super().__init__()
        self.model = InvaRNAForRegression(config=_model_config())

    def forward(self, input_ids, attention_mask=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits if hasattr(output, "logits") else output


def load_stage1(model_key: str, device):
    ckpt_path = STAGE1_CKPTS[model_key]
    state = _checkpoint_state(ckpt_path, device)
    normalized = {}
    for key, value in state.items():
        if key.startswith("model.model."):
            normalized[key.replace("model.model.", "model.", 1)] = value
        elif key.startswith("model."):
            normalized[key] = value
        else:
            normalized["model." + key] = value
    model = _Stage1Model()
    model.load_state_dict(normalized, strict=True)
    model.to(device).eval()
    print(f"[stage1] Loaded [{model_key}] from {ckpt_path}", flush=True)
    return model


def run_invarna_inference(model, df_all, rank, world_size, device, batch_size=32, phase="INFER"):
    missing = {"mrna", "utr5_size"} - set(df_all.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    tokenizer = InvaRNATokenizer(model_max_length=TOTAL_LENGTH)
    pad_id = tokenizer.pad_token_id
    unk_id = tokenizer.convert_tokens_to_ids("N")
    ids_to_mask = torch.tensor([pad_id, unk_id], device=device)
    df_all = df_all.reset_index(drop=True)
    shard = df_all.iloc[rank::world_size].copy()
    predictions = []
    iterator = range(0, len(shard), batch_size)
    if rank == 0:
        iterator = tqdm(iterator, desc=f"[{phase}]")

    with torch.no_grad():
        for start in iterator:
            batch = shard.iloc[start:start + batch_size]
            batch_ids = []
            for _, row in batch.iterrows():
                sequence = align_and_pad(row["mrna"], row["utr5_size"])
                ids = tokenizer.encode(sequence, add_special_tokens=False)
                batch_ids.append([pad_id if token == unk_id else token for token in ids])
            input_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
            attention_mask = (~torch.isin(input_ids, ids_to_mask)).long()
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                output = model(input_ids, attention_mask=attention_mask)
            output = output.logits if hasattr(output, "logits") else output
            values = output.squeeze(-1).float()
            predictions.extend([values.item()] if values.ndim == 0 else values.cpu().tolist())

    shard["_pred"] = predictions
    shard.to_parquet(f"_tmp_{phase}_rank{rank}.parquet")
    if world_size > 1:
        dist.barrier()
    result = None
    if rank == 0:
        parts = [pd.read_parquet(f"_tmp_{phase}_rank{index}.parquet") for index in range(world_size)]
        result = pd.concat(parts).sort_index()
        for index in range(world_size):
            Path(f"_tmp_{phase}_rank{index}.parquet").unlink(missing_ok=True)
    if world_size > 1:
        dist.barrier()
    return result


def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if str(path).endswith((".parquet", ".pq")) else pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description="InvaRNA Stage1/Stage2 TE inference")
    parser.add_argument("--model", required=True, choices=ALL_MODELS_WITH_CUSTOM)
    parser.add_argument("--ckpt", default=None, help="Checkpoint for --model stage2_ckpt")
    parser.add_argument("--pred_col", default=None)
    parser.add_argument("--input", required=True, help="Input parquet or CSV")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    if args.model == "stage2_ckpt" and not args.ckpt:
        parser.error("--ckpt is required when --model stage2_ckpt")
    rank, world_size, device = setup_dist()
    data = load_data(args.input)
    phase = args.model.upper()
    if args.model in STAGE2_CKPTS or args.model == "stage2_ckpt":
        model = load_stage2(args.model, device, ckpt_override=args.ckpt)
    else:
        model = load_stage1(args.model, device)
    result = run_invarna_inference(
        model, data, rank, world_size, device, batch_size=args.batch_size, phase=phase
    )
    if rank == 0:
        pred_col = args.pred_col or f"pred_{args.model}"
        result[pred_col] = result.pop("_pred")
        result.drop(columns=["mrna"], errors="ignore").to_csv(args.output, index=False)
        print(f"Saved {args.output} ({len(result)} rows)", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
