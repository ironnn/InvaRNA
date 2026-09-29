#!/usr/bin/env python
"""Export the exact seed-42 transcript batch used for the Fig. 2C routing panel.

The historical panel used the first shuffled batch from the backbone validation
FASTA with MLM enabled.  This exporter preserves both the original aligned FASTA
records and the post-MLM token IDs actually passed to the frozen backbone.  It
does not fabricate or resample sequences.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from invarna.models.tokenization import InvaRNATokenizer
from invarna.training.sampler import mlm_getitem


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gzip_text_writer(path: Path):
    """Return a deterministic UTF-8 gzip text stream (mtime and filename fixed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.open("wb")
    zipped = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return raw, zipped, io.TextIOWrapper(zipped, encoding="utf-8", newline="")


def read_indexed_fasta(
    path: Path, wanted_indices: set[int]
) -> tuple[int, dict[int, tuple[str, str]]]:
    """Read selected records with the standard library, avoiding a legacy .fxi index."""
    selected: dict[int, tuple[str, str]] = {}
    index = -1
    header: str | None = None
    sequence: list[str] = []

    def finish_record() -> None:
        if header is not None and index in wanted_indices:
            selected[index] = (header, "".join(sequence))

    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish_record()
                index += 1
                header = line[1:]
                sequence = []
            else:
                sequence.append(line)
        finish_record()
    record_count = index + 1
    missing = wanted_indices - set(selected)
    if missing:
        raise RuntimeError(f"FASTA lacks requested record indices: {sorted(missing)}")
    return record_count, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "assets/manuscript_figures/fig2_compact/F2C_sample_manifest_seed42.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "assets/manuscript_figures/fig2_compact"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=10000)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    args = parser.parse_args()

    expected = pd.read_csv(args.manifest).sort_values("batch_order")
    tokenizer = InvaRNATokenizer(model_max_length=args.max_length)
    expected_indices = expected["dataset_index_zero_based"].astype(int).tolist()
    record_count, records = read_indexed_fasta(args.fasta, set(expected_indices))

    # Mirror torch DataLoader(shuffle=True, generator=None, num_workers=0): iterator
    # creation consumes a base seed, RandomSampler consumes a second global seed,
    # and randperm itself uses a private generator. The surviving manifest provides
    # an independent assertion that this reconstruction is exact.
    torch.manual_seed(args.seed)
    _base_seed = int(torch.empty((), dtype=torch.int64).random_().item())
    sampler_seed = int(torch.empty((), dtype=torch.int64).random_().item())
    generator = torch.Generator().manual_seed(sampler_seed)
    sample_indices = torch.randperm(record_count, generator=generator)[: args.batch_size].tolist()
    if sample_indices != expected_indices:
        raise RuntimeError("sampled FASTA indices differ from the archived Fig. 2C manifest")
    observed_ids = [records[index][0] for index in sample_indices]
    if observed_ids != expected["fasta_record_id"].tolist():
        raise RuntimeError("sampled FASTA identifiers differ from the archived Fig. 2C manifest")

    masked_rows = []
    for index in sample_indices:
        sequence = records[index][1]
        token_ids = tokenizer(
            sequence,
            padding="max_length",
            max_length=args.max_length,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]
        token_ids.append(tokenizer.sep_token_id)
        tokens = torch.tensor(token_ids, dtype=torch.long)
        tokens = torch.where(
            tokens == tokenizer._vocab_str_to_int["N"],
            torch.tensor(tokenizer.pad_token_id),
            tokens,
        )
        masked, _target = mlm_getitem(
            tokens,
            mlm_probability=args.mlm_probability,
            contains_eos=True,
            tokenizer=tokenizer,
        )
        masked_rows.append(masked)
    input_ids = torch.stack(masked_rows)
    if tuple(input_ids.shape) != (args.batch_size, args.max_length):
        raise RuntimeError(f"unexpected masked-input shape: {tuple(input_ids.shape)}")

    fasta_output = args.output_dir / "F2C_seed42_sequences.fasta.gz"
    raw, zipped, text = gzip_text_writer(fasta_output)
    try:
        for record_id in observed_ids:
            record_index = sample_indices[observed_ids.index(record_id)]
            sequence = records[record_index][1]
            text.write(f">{record_id}\n")
            for offset in range(0, len(sequence), 80):
                text.write(sequence[offset : offset + 80] + "\n")
    finally:
        text.close()
        zipped.close()
        raw.close()

    token_output = args.output_dir / "F2C_seed42_masked_input_ids.tsv.gz"
    raw, zipped, text = gzip_text_writer(token_output)
    try:
        writer = csv.writer(text, delimiter="\t", lineterminator="\n")
        writer.writerow(["batch_order", "fasta_record_id", "input_ids"])
        for order, (record_id, row) in enumerate(zip(observed_ids, input_ids.tolist()), 1):
            writer.writerow([order, record_id, " ".join(map(str, row))])
    finally:
        text.close()
        zipped.close()
        raw.close()

    metadata = {
        "batch_size": args.batch_size,
        "dataloader_base_seed": _base_seed,
        "dataloader_sampler_seed": sampler_seed,
        "fasta_record_count": record_count,
        "fasta_sha256": sha256(args.fasta),
        "fixture_files": {
            fasta_output.name: sha256(fasta_output),
            token_output.name: sha256(token_output),
        },
        "max_length": args.max_length,
        "mlm_probability": args.mlm_probability,
        "pad_token_id": tokenizer.pad_token_id,
        "mask_token_id": tokenizer.mask_token_id,
        "sampling_seed": args.seed,
        "sampling_rule": "first DataLoader batch; shuffle=True; num_workers=0",
        "source_path_distributed": False,
    }
    metadata_output = args.output_dir / "F2C_fixture_metadata.json"
    metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        f"PASS: exported {args.batch_size} exact Fig. 2C records and masked inputs "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
