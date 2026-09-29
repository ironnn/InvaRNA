#!/usr/bin/env python
"""Publication smoke test: TDC, benchmark, model inference, and routing."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

from test_relative_paths import main as relative_path_checks

from invarna.data.fixed_frame import align_and_pad
from invarna.distillation.tdc import construct_labels
from invarna.evaluation.multi_species import summarize
from invarna.interpretability.routing import capture_forward_routing, summarize_routing
from invarna.synthetic.matched_random import mutate_matched


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def smoke_data_checks():
    smoke = ROOT / "assets/smoke"
    manifest = smoke / "MANIFEST.tsv"
    rows = list(csv.DictReader(manifest.open(), delimiter="\t"))
    if not rows:
        raise RuntimeError("empty smoke-data manifest")
    for row in rows:
        path = ROOT / row["path"]
        if not path.is_file() or _sha256(path) != row["sha256"]:
            raise RuntimeError(f"missing or modified smoke input: {row['path']}")

    student_required = {
        "transcript_id", "utr5_size", "cds_size", "utr3_size",
        "mrna", "species", "gene_id", "mean_te",
    }
    for path in sorted((smoke / "student").glob("*.parquet")):
        frame = pd.read_parquet(path)
        if len(frame) != 3 or not student_required.issubset(frame.columns):
            raise RuntimeError(f"invalid student smoke table: {path}")

    fm = pd.read_parquet(smoke / "synthetic/flow_matching_absolute_teacher.parquet")
    fm_steps = fm["transcript_id"].str.extract(r"_mut(\d+)$", expand=False).astype(int)
    if (
        len(fm) != 3
        or not fm_steps.between(201, 215).all()
        or not fm["data_type"].eq("train_mut").all()
        or not fm["soft_label_org"].equals(fm["pred_score"])
    ):
        raise RuntimeError("invalid absolute-teacher Flow-Matching smoke rows")

    feature_names = json.loads(
        (ROOT / "assets/checkpoints/teacher/metadata/feature_names.json").read_text()
    )
    for name in ("train.csv.gz", "validation.csv.gz"):
        frame = pd.read_csv(smoke / "teacher" / name)
        if list(frame.columns[-len(feature_names):]) != feature_names:
            raise RuntimeError(f"teacher feature order mismatch: {name}")

    for path in (smoke / "pretrain").glob("*.fasta"):
        if sum(line.startswith(">") for line in path.read_text().splitlines()) != 2:
            raise RuntimeError(f"invalid pretraining FASTA excerpt: {path}")

    with gzip.open(smoke / "design/ngf_rl_trajectory.csv.gz", "rt") as handle:
        ngf = pd.read_csv(handle)
    if len(ngf) != 3 or not {"sequence", "score"}.issubset(ngf.columns):
        raise RuntimeError("invalid NGF trajectory excerpt")
    print(f"PASS {len(rows)} checksummed real-data smoke inputs")


def fig2_source_checks():
    source = ROOT / "assets/manuscript_figures/fig2_compact"
    required = source / "F2A_left_species_UMAP.csv"
    if not required.is_file():
        print("SKIP full Fig. 2 source checks (install the external figure-data archive)")
        return
    entries = []
    for line in (source / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        expected, filename = line.split("  ", 1)
        path = source / filename
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"missing or modified Fig. 2 source: {filename}")
        entries.append(filename)
    umap = pd.read_csv(source / "F2A_left_species_UMAP.csv")
    manifest = pd.read_csv(source / "F2C_sample_manifest_seed42.csv")
    if len(umap) != 7000 or not umap["display_species"].value_counts().eq(1000).all():
        raise RuntimeError("invalid Fig. 2A compact source table")
    if len(manifest) != 24 or manifest["batch_order"].tolist() != list(range(1, 25)):
        raise RuntimeError("invalid Fig. 2C seed-42 manifest")
    print(f"PASS {len(entries)} checksummed Fig. 2 source/fixture files")


def core_checks():
    relative_path_checks()
    smoke_data_checks()
    fig2_source_checks()
    short = "A" * 12
    assert align_and_pad(short, 4).startswith("N" * 996 + short)
    long = "C" * 1001 + "ATG"
    assert align_and_pad(long, 1001).startswith(long)
    rows = [
        ("g1_mut0", "AAAA", 1.0, 1.5, "wt"),
        ("g1_mut1", "AAAT", 1.2, 1.5, "k80"),
        ("g1_mut2", "AATT", 1.4, 1.5, "k80"),
        ("g1_mut201", "TTTT", 2.0, 1.5, "flow_matching"),
        ("g1_mut3", "AATA", 0.8, 1.5, "matched_random"),
    ]
    frame = pd.DataFrame(rows, columns=["transcript_id", "mrna", "pred_score", "mean_te", "variant_source"])
    labels = construct_labels(frame, alpha=1.0).set_index("transcript_id")
    assert labels.loc["g1_mut0", "training_label"] == 1.5
    assert abs(labels.loc["g1_mut1", "training_label"] - 1.6) < 1e-9
    assert labels.loc["g1_mut201", "training_label"] == 2.0
    assert labels.loc["g1_mut3", "training_label"] == 0.8
    benchmark = pd.DataFrame({"species": ["a", "a", "a", "b", "b", "b"], "truth": [0, 1, 2, 0, 2, 1], "prediction": [0, 1, 2, 0, 2, 1]})
    scores = summarize(benchmark, "truth", "prediction")
    assert len(scores) == 2 and (scores["spearman"] == 1).all()
    wt_sequence = "AAAA" + "ATGGCTGCTTAA" + "CCCC"
    matched = mutate_matched(wt_sequence, 4, 12, 4, 2, 2, 1, seed=42)
    assert sum(a != b for a, b in zip(wt_sequence[:4], matched[:4])) == 2
    assert sum(a != b for a, b in zip(wt_sequence[4:16], matched[4:16])) == 2
    assert sum(a != b for a, b in zip(wt_sequence[16:], matched[16:])) == 1
    assert matched[4:7] == wt_sequence[4:7] and matched[13:16] == wt_sequence[13:16]
    print("PASS checkpoint-compatible Stage-2 sequence framing")
    print("PASS TDC label construction (K80/FM/matched rules)")
    print("PASS region-wise matched-random burden and CDS boundary protection")
    print("PASS benchmark metric entry point")


def model_checks(device):
    from invarna.evaluation.inference_api import InvaRNAPredictor
    predictor = InvaRNAPredictor("final_tdc", device=device)
    sequence = "ACGTACGTACGT" + "ATG" + "GCC" * 20 + "TAA" + "ACGT" * 10
    prediction, routing = capture_forward_routing(
        predictor.model,
        lambda: predictor.predict([sequence], utr5_sizes=[12], batch_size=1),
    )
    if len(prediction) != 1 or not torch.isfinite(torch.tensor(prediction)).all():
        raise RuntimeError("non-finite single-sequence prediction")
    valid = routing[:, 988:988 + len(sequence), :]
    score = summarize_routing(valid)
    if score.numel() != 4 or not torch.isfinite(score).all():
        raise RuntimeError("invalid routing score")
    print(f"PASS checkpoint load and one-sequence inference: {prediction[0]:.6f}")
    print("PASS routing score: " + ", ".join(f"{value:.6f}" for value in score.tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--core-only", action="store_true", help="Skip checkpoint-dependent model checks")
    args = parser.parse_args()
    core_checks()
    if not args.core_only:
        model_checks(args.device)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
