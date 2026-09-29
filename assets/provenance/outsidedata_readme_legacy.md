# Local benchmark assets (not uploaded to GitHub)

This directory is the self-contained, local-only asset bundle for reproducing the
external-LLM TE benchmark.  Git ignores its contents except this README and
`MANIFEST.tsv`.  Do not use `git add -f` on the binary assets.

Tiny real-row excerpts of the common TE splits and teacher/reference tables are
tracked separately under `assets/smoke/`. They validate schemas and loaders only; the
full files in this directory remain required for training and manuscript metrics.

Layout:

```text
assets/
├── benchmark_data/pure_te_bench/
│   ├── human_train_wt.parquet
│   ├── mouse_train_wt.parquet
│   ├── human_val_wt.parquet
│   └── human_test_wt.parquet
├── fig3_panel_g/
│   ├── input/                  # exact 100k/library source and derived 3k panel input
│   └── predictions/            # exact manuscript predictions for all three models
├── fig2/
│   ├── f2a/                    # seven-species metadata and 1,024-d embeddings
│   └── f2b/                    # MRL/HL/EL labels, cached embeddings, code, results
├── figure_data/
│   ├── fig3_submission/        # superseded July 17 plotting tables
│   ├── fig3_current_nm/        # canonical current nm.pdf plotting/source package
│   ├── fig3_0522/              # historical-only May 22 combined figure
│   ├── fig4_submission/        # final A-G tables plus recovered older H values
│   └── fig5/                   # plotting tables and raw experimental/RL inputs
├── privileged_teacher/
│   ├── reference/              # human/mouse half-life and bidirectional RBH tables
│   └── training/               # exact prepared human/mouse teacher feature tables
├── synthetic_generation/
│   └── anchors/                # human/mouse WT anchor tables
├── species_ablation/
│   ├── data/                   # exact three WT training conditions
│   ├── original_assets/checkpoints/   # immutable source copies for nine selected runs
│   ├── run_metrics/            # complete per-epoch Lightning metrics
│   └── selected_predictions/   # validation/test predictions at selected epochs
└── assets/checkpoints/
    ├── pretrained/models/     # upstream model snapshots/weights
    └── te_finetuned/          # best-by-validation-R2 TE checkpoints + licenses
```

Every copied external pretrained-model tree and every third-party-derived fine-tuned
checkpoint directory retains its applicable license/NOTICE. These artifacts are not
covered by the repository MIT license. The authoritative inventory is
`external_models/THIRD_PARTY_LICENSES.tsv`; CodonBERT and RiboNN are restricted to
academic/non-commercial use and cannot enter an unrestricted public bundle without
permission. Validate the repository plus this archive with:

```bash
python external_models/audit_licenses.py
```

The four benchmark tables are gene-disjoint human train/validation/test plus the
mouse training set used by the completed runs.  Validation selects checkpoints;
the human test set is evaluation-only.

`species_ablation/` closes the three-condition, three-seed human/mouse WT comparison.
All conditions use the same human-only validation/test splits, and the selected epoch
is always the maximum validation R². Run
`python benchmarks/rbh_sensitivity/audit.py` to check composition, gene
disjointness, RBH removal, epoch selection, checkpoint identity, and reported R².

`privileged_teacher/` closes the local dependency set for the deployed teacher:
the engineered-feature extractor and production scripts are tracked in
`invarna/distillation/privileged_teacher_source/`, while the large/licensed tables
remain here. `synthetic_generation/anchors/` retains the WT tables required by the
K80/matched label workflows. These assets are present locally but still need a stable
anonymous archival URL and redistribution approval before external review.

`fig3_panel_g/` is the complete local data bundle for the visible Fig. 3g MPRA
bar. It retains the original 1,000,000-row processed input, its exact deterministic
30,000-row panel subset, the original RiboNN and UTR-LM 1,000,000-row prediction
files used for the current panel (retained from July 17), and the separate 30,000-row checkpoint-rerun
tables. Run `python figures/fig3/fig3g_mpra.py` to verify their hashes and
regenerate the canonical panel.

`fig2/` retains the large source assets behind the compact Fig. 2 tables:
the seven-species embeddings and the complete cached embedding/label inputs used for
the mean-ribosome-load, half-life, and expression-level LightGBM probes. The exact
24-sequence F2C fixture and finalized plotting tables are installed from the external
archive under `assets/manuscript_figures/fig2_compact/`; the backbone checkpoint is under
`assets/checkpoints/backbone/`.

`figure_data/` contains the recovered F3-F5 plotting/source inputs. The current
2026-08-05 `nm.pdf` package is canonical; July 17 and May 22 are retained as
historical-only. Use `docs/FIGURE_AUDIT.csv` to see which panels
are fully traced, hardcoded, or missing raw provenance.

The selected Evo2 full-parameter run is `evo2_8k`: first 8,000 nt, current
`nm.pdf` comparison, test R2 0.730300.

The manifest records sizes and SHA-256 hashes.  To validate a separately supplied
bundle, run:

```bash
python tests/verify_assets.py
```

To recompute the reported R2 values from the selected checkpoints' saved test
predictions (no GPU required), run:

```bash
python benchmarks/heldout_human_te/verify_external_models.py
```

The large files cannot be included in the anonymous GitHub repository using normal
Git.  In particular, the Evo2 and LucaOne files exceed GitHub's ordinary file-size
limit and common Git-LFS per-file limits.  The anonymous-review release therefore
needs a separate checkpoint/data archive whose extracted root is `assets/`.
That archive must also preserve `docs/THIRD_PARTY_LICENSES.md`, the license inventory, and
the license files adjacent to each external checkpoint.
