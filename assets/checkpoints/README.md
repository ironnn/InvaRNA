# Checkpoints

Git tracks exactly seven individually compressed InvaRNA-native weights under
this directory. A checkout contains all seven `.zst` files (500,739,484 bytes,
477.54 MiB); decompressing them creates seven ignored runtime files
(583,471,695 bytes, 556.44 MiB). No separate checkpoint site or package is
needed. The public Fig. 2--5 redraws use frozen numeric source tables and do
not load weights.

The **minimal native checkpoint set** has exactly seven files after decompression:
`backbone/pretrained_step13500.pt` for embedding extraction
and backbone-routing replay; `teacher/privileged_teacher.pkl`
for teacher scoring; `te_student/final_tdc.ckpt` for deployed TE inference and
model smoke; `te_student/w0.ckpt`, `w1.ckpt`, `w2.ckpt`, and
`half_life/final_half_life.ckpt` for the updated Fig. 5 TE/HL design objective.
The five teacher metadata JSON files are also in this Git repository.
Figure-only redraws and CPU-only smoke tests need none of these seven binaries.
Historical teacher input tables are not distributed; only the fitted teacher model
and its metadata are released. Stage-2 TE/HL checkpoints are self-contained and do
not separately load the backbone weight file.

The w0/w1/w2 names map
to the former `wt`/`world_new`/`world_new07522` files; their contents and checksums
are unchanged. The public model keys are also `w0`, `w1`, and `w2`.

From the repository root, verify the compressed files, decompress them, then
verify the exact runtime bytes (requires `zstd`):

```bash
(cd assets/checkpoints && sha256sum -c COMPRESSED_SHA256SUMS)
find assets/checkpoints -type f -name '*.zst' -print0 | \
  while IFS= read -r -d '' archive; do zstd -dk "$archive"; done
(cd assets/checkpoints && sha256sum -c REQUIRED_SHA256SUMS)
python tests/smoke/test_pipeline.py --device cuda:0
```

For example, `te_student/w0.ckpt.zst` becomes `te_student/w0.ckpt` in place.
The uncompressed files are ignored by Git. To keep them elsewhere, decompress
the same directory tree there and set `INVARNA_CHECKPOINT_ROOT` to that tree.

`COMPRESSED_SHA256SUMS` lists only the seven tracked archives;
`REQUIRED_SHA256SUMS` lists the corresponding seven uncompressed files.
`SHA256SUMS`, `INFERENCE_SHA256SUMS`, `ABLATION_SHA256SUMS`, and
`SPECIES_ABLATION_SHA256SUMS` are retained as historical/full-inventory integrity
records; they are **not** the minimal package manifest. The optional Fig. 3
ablation, nine-species-ablation, Stage-1, and legacy-registry checkpoints remain
in a private non-release archive. They are not part of the public upload.
Third-party baseline weights are obtained from their original distributors, not
repacked into this native package.

`python tests/verify_assets.py` checks sizes of all tracked assets, including
the compressed weights; `--full` hashes them too. Full benchmark and training
inputs are not supplied with the seven weights. External-model download sources, versions, licenses,
and SHA-256 values are recorded under `external_models/`.
