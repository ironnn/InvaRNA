# External-model TE benchmark

This directory contains the actual code used to full-parameter fine-tune the external
sequence models on the common TE benchmark.  It includes the unified loading/training
implementation, model-specific training entry points, and the required upstream runtime
code. Upstream licenses are retained with their source trees. The repository-level MIT
license does not apply to copied third-party code, model artifacts, or their derivatives;
see [`THIRD_PARTY_LICENSES.tsv`](THIRD_PARTY_LICENSES.tsv) and
[`../docs/THIRD_PARTY_LICENSES.md`](../docs/THIRD_PARTY_LICENSES.md).

The copied upstream trees retain third-party author attribution. Machine-specific paths
in copied Orthrus/UTR-LM examples and defaults were changed mechanically to repository-
relative paths; UTR-LM requirement entries that pointed to local Conda build directories
were reduced to package names. No model or training algorithm was changed by this
portability cleanup. CodonBERT and RiboNN carry academic/non-commercial restrictions;
the strict release audit fails until those artifacts are excluded from an unrestricted
public bundle or explicit redistribution permission is obtained.

Large training data are not uploaded to GitHub; seven compressed native
checkpoints are under `assets/checkpoints/` in Git. External baseline weights
are **not** part of this repository. Download them only when rerunning an external-model
benchmark, using the exact source/version/license records in `DOWNLOADS.tsv` and the
per-file checksums in `CHECKSUMS.tsv`. The plotting workflows use committed prediction
tables and therefore do not require these weights.
All commands below are run from the repository root.

Before packaging copied models, run:

```bash
python external_models/audit_licenses.py
python external_models/audit_licenses.py --strict-public
```

The first command validates the inventory and local license evidence. The second is an
unrestricted-public-release gate and is expected to fail while restricted CodonBERT or
RiboNN artifacts remain in scope.

The shared benchmark split contains human train/validation/test and mouse train tables.
Every run uses seed 2222, selects its checkpoint by validation R2, and reports the untouched
human test R2.  Model regions and context lengths are recorded in
`benchmarks/configs/external_llm_te.yaml`.

Examples:

```bash
# Recompute every saved held-out R2 without loading large models.
python benchmarks/heldout_human_te/verify_external_models.py

# Re-run a selected checkpoint on the untouched human test split.  Use the
# model-specific environment listed in external_models/common/backbones.py.
python external_models/inference/evaluate_te_checkpoint.py \
  --model codonbert \
  --output outputs/codonbert_human_test.csv

# Full-parameter training examples (expensive).
python external_models/training/train_te_codonbert.py \
  --data_dir assets/training_data/common_te_splits \
  --output_dir outputs/te_codonbert --seed 2222

python external_models/training/train_te_evo2.py \
  --data_dir assets/training_data/common_te_splits \
  --max_len 8000 --epochs 10 --grad_accum 16 \
  --output_dir outputs/te_evo2_8k --seed 2222
```

The 2,048-nt Evo2 run produced the value used in the manuscript comparison.  The later
8,000-nt experiment is retained separately and is not substituted into that table.

`DOWNLOADS.tsv` records the artifact-level source and version information. Where an
upstream repository or model card did not expose an immutable revision in the recovered
run, the table says so explicitly; do not silently substitute a newer checkpoint.

For a rerun, download the exact artifact from the URL in `DOWNLOADS.tsv`, pin an
immutable upstream commit/revision where the table requests it, place files under the
recorded `local_layout`, and compare every downloaded binary with `CHECKSUMS.tsv`.
For Hugging Face artifacts the corresponding pattern is:

```bash
hf download <repo-id> --revision <immutable-revision> --local-dir <local-layout>
```

The checksum table is a record for on-demand downloads, not a request to add those
weights to Git or to the publication asset archive. During this packaging update,
files larger than 1 GB were not rehashed; their SHA-256 values were carried forward
from the authenticated prior manifest, while sizes and presence were checked.

For the expensive Evo2 inference runs, use `--model evo2_8k`; the command loads the
selected checkpoint and context length. Evo2
uses its original single-process, multi-GPU pipeline sharding and must not be launched
with `torchrun`.
