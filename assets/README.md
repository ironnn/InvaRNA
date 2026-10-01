# Frozen publication assets

This directory contains the minimal frozen data needed to redraw the published
Fig. 2--5 panels. Code that creates new synthetic sequences or labels lives
separately under `data_generation/`.

The Git repository contains this plotting asset set, the 1,115-row human WT
validation and test tables under `benchmark_data/human_te/`, manifests,
documentation, seven individually compressed native weights, and smoke examples.
Supplementary Data 1 and 2 are uploaded independently and are not part of this
Git repository. External baseline model
weights are excluded; see `../external_models/` for download sources, versions,
licenses, and SHA-256 records.

```text
assets/
├── benchmark_data/human_te/  # human WT validation/test tables for direct R² checks
├── training_data/       # tiny pretraining smoke fixture only
├── checkpoints/         # seven compressed native weights plus integrity records
├── manuscript_figures/  # current frozen plotting source tables only
├── smoke/               # tiny real-row fixtures committed to Git
├── provenance/          # historical manifests and non-canonical model-name mapping
├── MANIFEST.tsv
└── SHA256SUMS
```

The seven native weights are tracked as `.zst` files; see `checkpoints/README.md`
for integrity checks and decompression. Full training corpora, other benchmark
inputs, raw design trajectories, and historical figure packages are not in Git.
They are not required by the public plotting scripts.

Verify the Git-tracked assets without running any model:

```bash
python tests/verify_assets.py
```

The original 274-species Ensembl mature-mRNA corpus is not redistributed. Download
the relevant Ensembl releases to rebuild it; model-training and full inference
assets other than the seven native weights are not part of this checkout.
