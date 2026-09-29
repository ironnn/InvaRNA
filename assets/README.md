# Frozen publication assets

This directory contains the minimal frozen data needed to redraw the published
Fig. 2--5 panels. Code that creates new synthetic sequences or labels lives
separately under `data_generation/`.

The Git repository contains only manifests, documentation, and kilobyte-scale smoke
examples. The native publication asset archive is distributed separately from GitHub.
External baseline model weights are deliberately excluded from that archive; see
`../external_models/` for download sources, versions, licenses, and SHA-256 records.

```text
assets/
├── training_data/       # tiny pretraining smoke fixture only
├── checkpoints/         # checksum/provenance records; no model binaries
├── manuscript_figures/  # current frozen plotting source tables only
├── smoke/               # tiny real-row fixtures committed to Git
├── provenance/          # historical manifests and non-canonical model-name mapping
├── MANIFEST.tsv
└── SHA256SUMS
```

The full training corpora, benchmark archives, raw design trajectories, historical
figure packages, and all checkpoint binaries are kept in a separate private,
non-release archive. They are not required by the public plotting scripts.

Verify installed assets without running any model:

```bash
python tests/verify_assets.py
```

The original 274-species Ensembl mature-mRNA corpus is not redistributed. Download
the relevant Ensembl releases to rebuild it; model-training and full inference
assets are maintained separately from this figure-only bundle.
