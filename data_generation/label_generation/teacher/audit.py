#!/usr/bin/env python
"""Audit the frozen privileged-teacher model, inputs, split, and RBH exposure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "assets/checkpoints/teacher/privileged_teacher.pkl"
METADATA = ROOT / "assets/checkpoints/teacher/metadata"
TRAINING = ROOT / "assets/training_data/teacher/full_inputs"
REFERENCE = ROOT / "assets/training_data/half_life/teacher_reference"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def base_transcript(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"_mut\d+$", "", regex=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-missing-local-assets",
        action="store_true",
        help="audit the tracked model/metadata even when the external asset archive is unavailable",
    )
    args = parser.parse_args()

    expected_model_hash = (
        "293ba0b0f2b8d792a36e2f6cde20f24adb7422ab3df8afc79667f8feb1b5dc06"
    )
    if sha256(MODEL) != expected_model_hash:
        raise SystemExit("teacher checkpoint SHA-256 mismatch")

    model = joblib.load(MODEL)
    model_features = list(model.feature_name_)
    frozen_features = json.loads((METADATA / "feature_names.json").read_text())
    if model_features != frozen_features:
        raise SystemExit("model feature order differs from frozen feature_names.json")

    groups = {
        "total": len(model_features),
        "backbone_embedding": sum(name.startswith("feat_") for name in model_features),
        "half_life": sum(name == "hl" for name in model_features),
    }
    groups["engineered_sequence"] = (
        groups["total"] - groups["backbone_embedding"] - groups["half_life"]
    )
    if groups != {
        "total": 3274,
        "backbone_embedding": 3072,
        "half_life": 1,
        "engineered_sequence": 201,
    }:
        raise SystemExit(f"unexpected fitted feature composition: {groups}")

    required_source = [
        ROOT / "data_generation/label_generation/teacher/source/utils/lgbm_feature_extract_from_str.py",
        ROOT / "data_generation/label_generation/teacher/source/teacher_train.py",
        ROOT / "data_generation/label_generation/teacher/source/label_engineering_homosplit_v2.py",
    ]
    missing_source = [str(path.relative_to(ROOT)) for path in required_source if not path.is_file()]
    if missing_source:
        raise SystemExit(f"missing privileged-teacher source: {missing_source}")

    result: dict[str, object] = {
        "checkpoint_sha256": expected_model_hash,
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "features": groups,
        "source_snapshot": "present",
    }

    human_path = TRAINING / "train_ready_human_regional_strict_20260117_pred_8gpu_bf16.pkl"
    mouse_path = TRAINING / "train_ready_mouse_regional_strict_20260117_pred_8gpu_bf16.pkl"
    h2m_path = REFERENCE / "rbh_human_to_mouse.parquet"
    refs = [
        REFERENCE / "human_time.csv",
        REFERENCE / "mouse_time.csv",
        h2m_path,
        REFERENCE / "rbh_mouse_to_human.parquet",
        human_path,
        mouse_path,
    ]
    missing_local = [str(path.relative_to(ROOT)) for path in refs if not path.is_file()]
    if missing_local:
        result["local_assets"] = {"missing": missing_local}
        print(json.dumps(result, indent=2))
        if args.allow_missing_local_assets:
            return
        raise SystemExit("local privileged-teacher archive is incomplete")

    human = pd.read_pickle(human_path)
    mouse = pd.read_pickle(mouse_path)
    h2m = pd.read_parquet(h2m_path)
    human_time = pd.read_csv(REFERENCE / "human_time.csv")
    mouse_time = pd.read_csv(REFERENCE / "mouse_time.csv")
    split = json.loads((METADATA / "split_indices.json").read_text())
    train_idx = split["human_train_idx"]
    val_idx = split["human_val_idx"]
    test_idx = split["human_test_idx"]
    if (len(train_idx), len(val_idx), len(test_idx)) != (8917, 1115, 1115):
        raise SystemExit("unexpected frozen human split sizes")
    if set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx):
        raise SystemExit("frozen human split indices overlap")
    if len(set(train_idx) | set(val_idx) | set(test_idx)) != len(human):
        raise SystemExit("frozen human split does not cover the prepared human table exactly once")

    gene_sets = [
        set(human.iloc[index]["human_gene_id"].astype(str))
        for index in (train_idx, val_idx, test_idx)
    ]
    if gene_sets[0] & gene_sets[1] or gene_sets[0] & gene_sets[2] or gene_sets[1] & gene_sets[2]:
        raise SystemExit("frozen human split is not gene-disjoint")

    held = human.iloc[val_idx + test_idx]
    held_ids = set(base_transcript(held["transcript_id"]))
    h2m_map = dict(zip(h2m["tid"].astype(str), h2m["homolog_tid"].astype(str)))
    mapped_mouse = {h2m_map[tid] for tid in held_ids if tid in h2m_map}
    mouse_ids = base_transcript(mouse["transcript_id"])
    exposed_mouse_rows = int(mouse_ids.isin(mapped_mouse).sum())
    human_hl_genes = set(human_time["Ensembl Gene Id"].astype(str))
    mouse_hl_genes = set(mouse_time["Ensembl Gene Id"].astype(str))
    human_gene_ids = human["human_gene_id"].astype(str).str.split(".").str[0]
    mouse_gene_ids = mouse["gene_id_x"].astype(str).str.split(".").str[0]

    result["local_assets"] = {
        "human_teacher_rows": len(human),
        "mouse_teacher_rows": len(mouse),
        "rbh_pairs": len(h2m),
        "human_half_life": {
            "rows": len(human_time),
            "unique_genes": int(human_time["Ensembl Gene Id"].nunique()),
            "non_null_pc1": int(human_time["half-life (PC1)"].notna().sum()),
        },
        "mouse_half_life": {
            "rows": len(mouse_time),
            "unique_genes": int(mouse_time["Ensembl Gene Id"].nunique()),
            "non_null_pc1": int(mouse_time["half-life (PC1)"].notna().sum()),
        },
        "half_life_mapping": {
            "human_mapped": int(human_gene_ids.isin(human_hl_genes).sum()),
            "human_pred_score_fallback": int((~human_gene_ids.isin(human_hl_genes)).sum()),
            "mouse_mapped": int(mouse_gene_ids.isin(mouse_hl_genes).sum()),
            "mouse_pred_score_fallback": int((~mouse_gene_ids.isin(mouse_hl_genes)).sum()),
        },
        "human_split": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "human_split_gene_overlap": 0,
    }
    result["observed_cross_species_exposure"] = {
        "held_out_human_transcripts": len(held_ids),
        "held_out_human_transcripts_with_rbh": len(mapped_mouse),
        "corresponding_mouse_rows_used_by_teacher": exposed_mouse_rows,
        "interpretation": (
            "human validation/test rows are absent from training; mouse ortholog "
            "inclusion is a cross-species design factor assessed by a separate ablation"
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
