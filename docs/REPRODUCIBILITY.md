# Reproducibility guide

1. Create the pinned environment from `environment.yml` (or the Linux lock snapshot).
2. Download the separately deposited asset archive and extract it as `assets/`.
3. Run `python tests/verify_assets.py` and the third-party license audit.
4. Run `python tests/smoke/test_pipeline.py --core-only`; on an NVIDIA GPU, run the
   full `python tests/smoke/test_pipeline.py --device cuda:0` checkpoint/inference test.
5. Run the desired `figures/fig*/reproduce.sh` entry point.

All commands are run from the repository root, and every project data/model/output
reference is repository-relative. `python tests/test_relative_paths.py` checks both
authored runtime files and copied external-model source/configuration for machine-specific
absolute paths.

Expensive pretraining/SFT commands remain available, but manuscript verification uses
the frozen checkpoints and processed intermediates by default. Data-split leakage
controls are tested by `tests/test_data_splits.py`.
