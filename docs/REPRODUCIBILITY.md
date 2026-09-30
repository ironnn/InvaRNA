# Reproducibility guide

1. Create the pinned environment from `environment.yml` (or the Linux lock snapshot).
2. Verify the figure-only assets included in the GitHub checkout with
   `python tests/verify_assets.py` and `python external_models/audit_licenses.py`.
3. Run `python tests/smoke/test_pipeline.py --core-only`,
   `python tests/test_manuscript_values.py`, and the desired
   `figures/fig*/reproduce.sh` entry point.
4. For model inference, install the seven-file minimal native checkpoint package under
   `assets/checkpoints/`, run the `REQUIRED_SHA256SUMS` check in
   `assets/checkpoints/README.md`, then run
   `python tests/smoke/test_pipeline.py --device cuda:0` on an NVIDIA GPU.
5. Full teacher retraining and most benchmark reruns require historical input tables
   described in `docs/DATA_PROVENANCE.md` and `docs/BENCHMARKS.md`; these are not
   distributed with this release.

All commands are run from the repository root, and every project data/model/output
reference is repository-relative. `python tests/test_relative_paths.py` checks both
authored runtime files and copied external-model source/configuration for machine-specific
absolute paths.

Expensive pretraining/SFT commands remain available, but manuscript verification uses
the frozen checkpoints and processed intermediates by default. Data-split leakage
controls are tested by `tests/test_data_splits.py`.
