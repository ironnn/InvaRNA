#!/usr/bin/env python
"""Build tiny, deterministic smoke inputs by subsetting the real local datasets.

The complete inputs remain Git-ignored. This script never fabricates sequences,
labels, features, or split membership; it copies a few rows from the preserved
local sources and records their provenance in assets/smoke/MANIFEST.tsv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRETRAIN_ROOT = ROOT.parent / "InvaRNA_backbone_pretrain/pretrain/data"
DEFAULT_LOCAL_TRAINING = ROOT / "assets/training_data/human_te/local_training_tables"
DEFAULT_OUTSIDE = ROOT / "assets"
DEFAULT_FM = ROOT / "assets/training_data/synthetic/flow_matching"
DEFAULT_SCORED_SYNTHETIC = (
    ROOT / "assets/training_data/synthetic/intermediate/student_train_merged_final_v3.parquet"
)
OUTPUT = ROOT / "assets/smoke"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_parquet_rows(path: Path, count: int) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    batch = next(parquet.iter_batches(batch_size=count))
    return batch.to_pandas().head(count).reset_index(drop=True)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def read_fasta_records(path: Path, count: int) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence: list[str] = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(sequence)))
                    if len(records) == count:
                        break
                header, sequence = line, []
            else:
                sequence.append(line)
        else:
            if header is not None and len(records) < count:
                records.append((header, "".join(sequence)))
    if len(records) != count:
        raise ValueError(f"{path}: requested {count} FASTA records, found {len(records)}")
    return records


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for header, sequence in records:
            handle.write(header + "\n")
            for offset in range(0, len(sequence), 80):
                handle.write(sequence[offset : offset + 80] + "\n")


def relative_source(path: Path | tuple[Path, ...]) -> str:
    if isinstance(path, tuple):
        return ";".join(relative_source(item) for item in path)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return "../" + path.resolve().relative_to(ROOT.parent.resolve()).as_posix()


def teacher_frame(
    frame: pd.DataFrame,
    species: str,
    feature_names: list[str],
    half_life: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    result["species"] = species
    if species == "human":
        result["gene_id"] = result["human_gene_id"]
        result["mean_te"] = result["humanTE"]
    else:
        result["gene_id"] = result["gene_id_x"]
    clean_gene = result["gene_id"].astype(str).str.split(".").str[0]
    mapping = (
        half_life.drop_duplicates("Ensembl Gene Id")
        .set_index("Ensembl Gene Id")["half-life (PC1)"]
    )
    result["hl"] = clean_gene.map(mapping).fillna(result["pred_score"])
    metadata = [
        "transcript_id", "gene_id", "species", "mrna",
        "utr5_size", "cds_size", "utr3_size", "mean_te",
    ]
    missing = set(feature_names) - set(result.columns)
    if missing:
        raise ValueError(f"teacher source is missing fitted features: {sorted(missing)}")
    return result[metadata + feature_names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-root", type=Path, default=DEFAULT_PRETRAIN_ROOT)
    parser.add_argument("--local-training-root", type=Path, default=DEFAULT_LOCAL_TRAINING)
    parser.add_argument("--outside-root", type=Path, default=DEFAULT_OUTSIDE)
    parser.add_argument("--fm-root", type=Path, default=DEFAULT_FM)
    parser.add_argument(
        "--scored-synthetic-table", type=Path, default=DEFAULT_SCORED_SYNTHETIC
    )
    args = parser.parse_args()

    generated: list[tuple[Path, Path | tuple[Path, ...], str, int]] = []

    # Exact full-length records from the production pretraining FASTA files.
    for source_name, output_name in (
        ("train0606.fasta", "train.fasta"),
        ("val0606.fasta", "validation.fasta"),
    ):
        source = args.pretrain_root / source_name
        output = OUTPUT / "pretrain" / output_name
        records = read_fasta_records(source, 2)
        write_fasta(records, output)
        generated.append((output, source, "first 2 complete FASTA records", len(records)))

    # Common gene-disjoint TE tables used by native SFT and external-model tuning.
    common = args.outside_root / "training_data/common_te_splits"
    student_sources = {
        "human_train.parquet": common / "human_train_wt.parquet",
        "mouse_train.parquet": common / "mouse_train_wt.parquet",
        "human_validation.parquet": common / "human_val_wt.parquet",
        "human_test.parquet": common / "human_test_wt.parquet",
    }
    for name, source in student_sources.items():
        output = OUTPUT / "student" / name
        frame = first_parquet_rows(source, 3)
        write_parquet(frame, output)
        generated.append((output, source, "first 3 rows; original split retained", len(frame)))

    # Final TDC, K80-only, and matched-random training tables. The first 256 rows
    # contain a WT anchor and the numbered variants for that anchor.
    synthetic_sources = {
        "tdc_final.parquet": args.local_training_root / "train_lite_v2.parquet",
        "k80.parquet": args.local_training_root / "train_lite_k80.parquet",
        "matched_random.parquet": args.local_training_root / "train_lite_matched.parquet",
    }
    for name, source in synthetic_sources.items():
        batch = next(pq.ParquetFile(source).iter_batches(batch_size=256)).to_pandas()
        first_base = str(batch.iloc[0]["transcript_id"]).rsplit("_mut", 1)[0]
        suffixes = ("_mut0", "_mut1", "_mut2")
        transcript = batch["transcript_id"].astype(str)
        wanted = transcript.str.startswith(first_base + "_mut") & transcript.str.endswith(suffixes)
        frame = batch.loc[wanted].head(5).reset_index(drop=True)
        output = OUTPUT / "synthetic" / name
        write_parquet(frame, output)
        generated.append((output, source, "first anchor: WT and numbered variants", len(frame)))

    # Teacher-scored FM replacements used for student training. In the preserved
    # merged table they are mut201-mut215; soft_label_org is the absolute teacher
    # prediction and is the training target for this variant class.
    scored = pq.ParquetFile(args.scored_synthetic_table)
    scored_columns = [
        "transcript_id", "gene_id", "species", "data_type", "mut_step",
        "pred_score", "mean_te", "soft_label_org", "soft_label_taylor",
        "mrna", "utr5_size", "cds_size", "utr3_size",
    ]
    # The historical file has five row groups and places human FM rows in group 2.
    # Assert the semantic identifiers below so a changed source layout fails loudly.
    scored_batch = scored.read_row_group(2, columns=scored_columns).to_pandas()
    identifiers = scored_batch["transcript_id"].astype(str)
    is_fm = identifiers.str.extract(r"_mut(\d+)$", expand=False).astype(int).between(201, 215)
    scored_fm = scored_batch.loc[
        scored_batch["data_type"].eq("train_mut") & is_fm
    ].head(3).reset_index(drop=True)
    if len(scored_fm) != 3 or not scored_fm["soft_label_org"].equals(scored_fm["pred_score"]):
        raise ValueError("could not recover three absolute-teacher FM training rows")
    scored_fm_output = OUTPUT / "synthetic/flow_matching_absolute_teacher.parquet"
    write_parquet(scored_fm, scored_fm_output)
    generated.append(
        (
            scored_fm_output,
            args.scored_synthetic_table,
            "first 3 human train_mut rows with transcript suffix mut201-mut215",
            len(scored_fm),
        )
    )

    # One complete generated sequence per mutation setting from the real FM pools.
    fm_files = {
        "flow_matching_mut50.csv.gz": args.fm_root / "mut50/val_141740.csv",
        "flow_matching_mut100.csv.gz": args.fm_root / "mut100/val_138900.csv",
        "flow_matching_mut150.csv.gz": args.fm_root / "mut150/val_138900.csv",
        "flow_matching_mut200.csv.gz": args.fm_root / "mut200/val_138900.csv",
    }
    for name, source in fm_files.items():
        frame = pd.read_csv(source, nrows=1)
        output = OUTPUT / "synthetic" / name
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(
            output, index=False, compression={"method": "gzip", "mtime": 0}
        )
        generated.append((output, source, "first complete row", len(frame)))

    # Privileged-teacher samples retain all 3,274 fitted feature columns. CSV gzip
    # avoids Parquet's disproportionate metadata overhead for a few wide rows.
    teacher_training = args.outside_root / "training_data/teacher/full_inputs"
    teacher_reference = args.outside_root / "training_data/half_life/teacher_reference"
    feature_names = json.loads(
        (ROOT / "assets/checkpoints/teacher/metadata/feature_names.json").read_text()
    )
    split = json.loads((ROOT / "assets/checkpoints/teacher/metadata/split_indices.json").read_text())
    human_source = teacher_training / "train_ready_human_regional_strict_20260117_pred_8gpu_bf16.pkl"
    mouse_source = teacher_training / "train_ready_mouse_regional_strict_20260117_pred_8gpu_bf16.pkl"
    human = pd.read_pickle(human_source)
    mouse = pd.read_pickle(mouse_source)
    human_hl_source = teacher_reference / "human_time.csv"
    mouse_hl_source = teacher_reference / "mouse_time.csv"
    human_hl = pd.read_csv(human_hl_source)
    mouse_hl = pd.read_csv(mouse_hl_source)
    human_train = teacher_frame(
        human.iloc[split["human_train_idx"][:2]], "human", feature_names, human_hl
    )
    mouse_train = teacher_frame(mouse.iloc[:2], "mouse", feature_names, mouse_hl)
    teacher_train = pd.concat([human_train, mouse_train], ignore_index=True)
    teacher_validation = teacher_frame(
        human.iloc[split["human_val_idx"][:3]], "human", feature_names, human_hl
    )
    for name, frame, selection in (
        ("train.csv.gz", teacher_train, "first 2 frozen human-train rows plus first 2 mouse rows"),
        ("validation.csv.gz", teacher_validation, "first 3 frozen human-validation rows"),
    ):
        output = OUTPUT / "teacher" / name
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(
            output,
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        sources = (human_source, human_hl_source)
        if name == "train.csv.gz":
            sources = (human_source, mouse_source, human_hl_source, mouse_hl_source)
        generated.append((output, sources, selection, len(frame)))
    for name, source in (
        ("human_half_life.csv", human_hl_source),
        ("mouse_half_life.csv", mouse_hl_source),
    ):
        frame = pd.read_csv(source, nrows=3)
        output = OUTPUT / "teacher" / name
        frame.to_csv(output, index=False)
        generated.append((output, source, "first 3 rows", len(frame)))
    rbh_source = teacher_reference / "rbh_human_to_mouse.parquet"
    rbh = first_parquet_rows(rbh_source, 3)
    rbh_output = OUTPUT / "teacher/rbh_human_to_mouse.parquet"
    write_parquet(rbh, rbh_output)
    generated.append((rbh_output, rbh_source, "first 3 rows", len(rbh)))
    del human, mouse

    # External-evaluation inputs are smoke-only and never become training rows.
    mpra_source = args.outside_root / "benchmark_data/mpra/fig3g/input/mpra_panel_g_3k.parquet"
    mpra = pd.read_parquet(mpra_source).groupby("sample", sort=False).head(1).reset_index(drop=True)
    mpra_output = OUTPUT / "evaluation/mpra.parquet"
    write_parquet(mpra, mpra_output)
    generated.append((mpra_output, mpra_source, "first row per MPRA library", len(mpra)))

    rpfd_source = ROOT / "assets/benchmark_data/rpfdb/rpfd_multi_species.parquet"
    final_species = set(
        pd.read_csv(ROOT / "assets/manuscript_figures/results/rpfd_final.csv")["species"]
    )
    rpfd = pd.read_parquet(rpfd_source)
    rpfd = (
        rpfd.loc[rpfd["species"].isin(final_species)]
        .groupby("species", sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    rpfd_output = OUTPUT / "evaluation/rpfd_multi_species.parquet"
    write_parquet(rpfd, rpfd_output)
    generated.append((rpfd_output, rpfd_source, "first row per species", len(rpfd)))

    manifest = OUTPUT / "MANIFEST.tsv"
    with manifest.open("w") as handle:
        handle.write("path\tbytes\tsha256\trows_or_records\tsource\tselection\n")
        for output, source, selection, rows in sorted(generated):
            handle.write(
                f"{output.relative_to(ROOT).as_posix()}\t{output.stat().st_size}\t"
                f"{sha256(output)}\t{rows}\t{relative_source(source)}\t{selection}\n"
            )
    print(f"wrote {len(generated)} real-data smoke files and {manifest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
