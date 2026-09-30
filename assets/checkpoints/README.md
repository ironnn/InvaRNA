# Checkpoints

Model weights are not part of the figure-only asset bundle. The separate native
checkpoint package is intended to be installed under this directory, preserving
paths such as `assets/checkpoints/te_student/final_tdc.ckpt` and
`assets/checkpoints/backbone/pretrained_step13500.pt`. The download URL/DOI has not
yet been added to this repository; a checkout alone cannot run checkpoint inference.
The public Fig. 2--5 redraws use frozen numeric source tables and do not load weights.
If the checkpoint archive is kept outside the checkout, set
`INVARNA_CHECKPOINT_ROOT=/path/to/archive/assets/checkpoints` before running native
inference or the full model smoke; the archive layout must match the paths below.

The **minimal native checkpoint package** has exactly seven files (583,471,695
bytes, 556.44 MiB): `backbone/pretrained_step13500.pt` for embedding extraction
and backbone-routing replay; `teacher/privileged_teacher.pkl`
for teacher scoring; `te_student/final_tdc.ckpt` for deployed TE inference and
model smoke; `te_student/w0.ckpt`, `w1.ckpt`, `w2.ckpt`, and
`half_life/final_half_life.ckpt` for the updated Fig. 5 TE/HL design objective.
The five teacher metadata JSON files are already in this Git repository.
Figure-only redraws and CPU-only smoke tests need none of these seven binaries.
Historical teacher input tables are not distributed; only the fitted teacher model
and its metadata are released. Stage-2 TE/HL checkpoints are self-contained and do
not separately load the backbone weight file.

The w0/w1/w2 names map
to the former `wt`/`world_new`/`world_new07522` files; their contents and checksums
are unchanged. The public model keys are also `w0`, `w1`, and `w2`.

Verify the minimal package after installing it under `assets/checkpoints/`:

```bash
(cd assets/checkpoints && sha256sum -c REQUIRED_SHA256SUMS)
python tests/smoke/test_pipeline.py --device cuda:0
```

If the seven files are downloaded separately as `.zst`, decompress each into its
matching subdirectory and original filename (for example,
`te_student/w0.ckpt.zst` becomes `te_student/w0.ckpt`) before running the hash check.

`SHA256SUMS`, `INFERENCE_SHA256SUMS`, `ABLATION_SHA256SUMS`, and
`SPECIES_ABLATION_SHA256SUMS` are retained as historical/full-inventory integrity
records; they are **not** the minimal package manifest. The optional Fig. 3
ablation, nine-species-ablation, Stage-1, and legacy-registry checkpoints remain
in a private non-release archive. They are not part of the public upload.
Third-party baseline weights are obtained from their original distributors, not
repacked into this native package.

`python tests/verify_assets.py` checks only the figure-only package and cannot
validate checkpoint binaries. Full benchmark and training inputs are not supplied
with the seven-weight package. External-model download sources, versions, licenses,
and SHA-256 values are recorded under `external_models/`.
