#!/usr/bin/env python
"""Audit the three-seed human/mouse WT ablation without retraining.

The audit recomputes R² from saved predictions, verifies best-validation epoch
selection, checks split/species composition and gene disjointness, and loads all
released canonical checkpoints through the compatibility serialization alias.
"""

from __future__ import annotations

import hashlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import ttest_rel


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets/benchmark_data/rbh_sensitivity/full"
RESULTS = ROOT / "assets/manuscript_figures/results"
SEEDS = (22222, 3407, 9713)
sys.path.insert(0, str(ROOT / "src"))


@dataclass(frozen=True)
class Run:
    condition: str
    seed: int
    epoch: int

    @property
    def stem(self) -> str:
        return f"{self.condition}_seed{self.seed}"

    @property
    def config(self) -> Path:
        return ROOT / "sft" / "configs" / "species_ablation" / f"{self.stem}.yaml"

    @property
    def checkpoint(self) -> Path:
        return ROOT / "assets" / "checkpoints" / "ablations" / "rbh_sensitivity" / f"{self.stem}.ckpt"


RUNS = (
    Run("human", 22222, 12),
    Run("human", 3407, 18),
    Run("human", 9713, 20),
    Run("human_mouse_rbh_filtered", 22222, 15),
    Run("human_mouse_rbh_filtered", 3407, 35),
    Run("human_mouse_rbh_filtered", 9713, 20),
    Run("human_mouse_full", 22222, 42),
    Run("human_mouse_full", 3407, 29),
    Run("human_mouse_full", 9713, 42),
)

EXPECTED_COMPOSITION = {
    "human_wt.parquet": {"human": 8917},
    "human_mouse_rbh_filtered_wt.parquet": {"human": 8917, "mouse": 9661},
    "human_mouse_full_wt.parquet": {"human": 8917, "mouse": 11433},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def r2(frame: pd.DataFrame) -> float:
    target = frame["target"].to_numpy(dtype=np.float64)
    pred = frame["pred"].to_numpy(dtype=np.float64)
    return float(1.0 - np.square(target - pred).sum() / np.square(target - target.mean()).sum())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_hashes() -> dict[str, str]:
    manifest = ROOT / "assets" / "checkpoints" / "SPECIES_ABLATION_SHA256SUMS"
    hashes = {}
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        hashes[Path(rel).name] = digest
    return hashes


def audit_splits() -> None:
    frames = {}
    for filename, expected in EXPECTED_COMPOSITION.items():
        frame = pd.read_parquet(ASSETS / "data" / filename)
        observed = frame["species"].value_counts().to_dict()
        require(observed == expected, f"{filename}: species counts {observed} != {expected}")
        require(frame["transcript_id"].is_unique, f"{filename}: duplicate transcript_id")
        frames[filename] = frame

    val = pd.read_parquet(ROOT / "assets/benchmark_data/human_te/human_val_wt.parquet")
    test = pd.read_parquet(ROOT / "assets/benchmark_data/human_te/human_test_wt.parquet")
    for split_name, frame in (("validation", val), ("test", test)):
        require(len(frame) == 1115, f"{split_name}: expected 1115 rows")
        require(set(frame["species"]) == {"human"}, f"{split_name}: must be human only")

    human_train = frames["human_wt.parquet"]
    train_genes = set(human_train["gene_id"].astype(str).str.split(".").str[0])
    val_genes = set(val["gene_id"].astype(str).str.split(".").str[0])
    test_genes = set(test["gene_id"].astype(str).str.split(".").str[0])
    require(train_genes.isdisjoint(val_genes), "human train/validation gene overlap")
    require(train_genes.isdisjoint(test_genes), "human train/test gene overlap")
    require(val_genes.isdisjoint(test_genes), "human validation/test gene overlap")

    rbh = pd.read_parquet(
        ROOT / "assets/training_data/half_life/teacher_reference/rbh_human_to_mouse.parquet"
    )
    heldout_tids = set(
        pd.concat([val["transcript_id"], test["transcript_id"]])
        .astype(str).str.removesuffix("_mut0")
    )
    forbidden_mouse = set(rbh.loc[rbh["tid"].isin(heldout_tids), "homolog_tid"])
    filtered_mouse = frames["human_mouse_rbh_filtered_wt.parquet"].query(
        "species == 'mouse'"
    )["transcript_id"].astype(str).str.removesuffix("_mut0")
    require(set(filtered_mouse).isdisjoint(forbidden_mouse), "RBH-filtered mouse rows contain held-out homologs")


def install_checkpoint_alias() -> None:
    from invarna.models import configuration

    legacy_package = sys.modules.setdefault("backbone", types.ModuleType("backbone"))
    legacy_package.__path__ = []
    sys.modules["backbone.configuration"] = configuration


def audit_runs() -> pd.DataFrame:
    require({run.seed for run in RUNS} == set(SEEDS), "unexpected public seed set")
    config_files = sorted((ROOT / "sft/configs/species_ablation").glob("*.yaml"))
    require(len(config_files) == len(RUNS), "config directory must contain exactly nine runs")

    install_checkpoint_alias()
    hashes = expected_hashes()
    rows = []
    state_signature = None
    for run in RUNS:
        config = yaml.safe_load(run.config.read_text())
        require(config["seed"] == run.seed, f"{run.stem}: config seed mismatch")
        require(config["selection"]["selected_epoch"] == run.epoch, f"{run.stem}: config epoch mismatch")
        require(config["selection"]["metric"] == "val_r2_global", f"{run.stem}: wrong selection metric")

        metrics = pd.read_csv(ASSETS / "run_metrics" / f"{run.stem}.csv")
        best = metrics.dropna(subset=["val_r2_global"]).sort_values("val_r2_global").iloc[-1]
        require(int(best["epoch"]) == run.epoch, f"{run.stem}: selected epoch is not validation maximum")

        validation = pd.read_csv(ASSETS / "selected_predictions" / f"{run.stem}_validation.csv")
        test = pd.read_csv(ASSETS / "selected_predictions" / f"{run.stem}_test.csv")
        # The historical multi-GPU DistributedSampler padded the 1,115-row split
        # to a multiple of world size, so its exact saved files contain 1,116
        # predictions.  Preserve and audit the reported run rather than silently
        # changing its metric after publication packaging.
        devices = int(config["training"]["devices"])
        expected_prediction_rows = ((1115 + devices - 1) // devices) * devices
        require(
            len(validation) == len(test) == expected_prediction_rows,
            f"{run.stem}: prediction row count mismatch",
        )
        validation_r2, test_r2 = r2(validation), r2(test)
        require(np.isclose(validation_r2, best["val_r2_global"], atol=1e-6), f"{run.stem}: validation R2 mismatch")

        require(sha256(run.checkpoint) == hashes[run.checkpoint.name], f"{run.stem}: checkpoint hash mismatch")
        payload = torch.load(run.checkpoint, map_location="cpu", weights_only=False)
        require(int(payload["epoch"]) == run.epoch, f"{run.stem}: checkpoint epoch mismatch")
        require(len(payload["state_dict"]) == 357, f"{run.stem}: expected 357 state tensors")
        signature = {key: tuple(value.shape) for key, value in payload["state_dict"].items()}
        if state_signature is None:
            state_signature = signature
        else:
            require(signature == state_signature, f"{run.stem}: state key/shape signature mismatch")

        rows.append({
            "configuration": run.condition,
            "seed": run.seed,
            "selected_epoch": run.epoch,
            "validation_r2": validation_r2,
            "test_r2": test_r2,
            "checkpoint": run.checkpoint.relative_to(ROOT).as_posix(),
        })
        del payload
    return pd.DataFrame(rows)


def audit_committed_results(frame: pd.DataFrame) -> None:
    committed = pd.read_csv(RESULTS / "species_ablation_three_seed.csv")
    keys = ["configuration", "seed", "selected_epoch", "checkpoint"]
    require(frame[keys].equals(committed[keys]), "committed per-run metadata differs")
    require(np.allclose(frame["validation_r2"], committed["validation_r2"], atol=5e-12), "committed validation R2 differs")
    require(np.allclose(frame["test_r2"], committed["test_r2"], atol=5e-12), "committed test R2 differs")

    summary = frame.groupby("configuration", sort=False)["test_r2"].agg(
        n_seeds="count", test_r2_mean="mean", test_r2_sample_std="std"
    ).reset_index()
    committed_summary = pd.read_csv(RESULTS / "species_ablation_three_seed_summary.csv")
    require(summary[["configuration", "n_seeds"]].equals(committed_summary[["configuration", "n_seeds"]]), "committed summary metadata differs")
    require(np.allclose(summary["test_r2_mean"], committed_summary["test_r2_mean"], atol=5e-12), "committed means differ")
    require(np.allclose(summary["test_r2_sample_std"], committed_summary["test_r2_sample_std"], atol=5e-12), "committed standard deviations differ")

    pivot = frame.pivot(index="seed", columns="configuration", values="test_r2")
    comparisons = (
        ("human_mouse_full_vs_human", "human_mouse_full", "human"),
        ("human_mouse_full_vs_human_mouse_rbh_filtered", "human_mouse_full", "human_mouse_rbh_filtered"),
        ("human_mouse_rbh_filtered_vs_human", "human_mouse_rbh_filtered", "human"),
    )
    tests = []
    for name, left, right in comparisons:
        test = ttest_rel(pivot[left], pivot[right])
        tests.append((name, float((pivot[left] - pivot[right]).mean()), float(test.statistic), float(test.pvalue), len(SEEDS)))
    committed_tests = pd.read_csv(RESULTS / "species_ablation_three_seed_paired_tests.csv")
    observed_tests = pd.DataFrame(tests, columns=committed_tests.columns)
    require(observed_tests["comparison"].equals(committed_tests["comparison"]), "committed comparison order differs")
    for column in ("mean_paired_delta", "t_statistic", "p_value_two_sided"):
        require(np.allclose(observed_tests[column], committed_tests[column], atol=5e-12), f"committed {column} differs")

    for row in summary.itertuples(index=False):
        print(f"PASS {row.configuration}: n={row.n_seeds}, test R2={row.test_r2_mean:.6f} +/- {row.test_r2_sample_std:.6f}")


def main() -> None:
    audit_splits()
    frame = audit_runs()
    audit_committed_results(frame)
    print("PASS species-ablation audit: splits, configs, nine checkpoints, selection, and results")


if __name__ == "__main__":
    main()
