# Benchmarks

The package provides reusable human-TE inference, grouped multi-species metrics, MPRA
summaries, and representation-transfer metric utilities under `invarna/evaluation/`.
The manuscript representation-transfer probes used LightGBM, not ridge regression.
The recovered task-specific probe parameters, candidate-selection rule, split units,
and random seed are recorded in `benchmarks/configs/fig2.yaml` and
`benchmarks/configs/representation_transfer.yaml`.

The Git checkout contains the frozen Fig. 2--5 plotting tables, the human WT
validation/test splits, and seven compressed InvaRNA-native checkpoints. It does
not contain the full benchmark input and prediction archives, representation
embeddings, external baseline weights, or the optional ablation checkpoints.
Commands below that need those omitted files are explicitly marked; the main
`figures/fig*/reproduce.sh` entry points use the committed plotting tables.

## Fig. 2 frozen-representation benchmark

The compact finalized tables are included under
`assets/manuscript_figures/fig2_compact/` in the GitHub checkout. The full probe
labels and cached embeddings are not in Git. Reproduce the
submitted computational figure and validate all displayed values with:

```bash
python figures/fig2/reproduce.py
```

The half-life and expression-level candidate sweeps select one parameter set by mean
validation R2 across backbones, then apply it unchanged to every backbone in that
task. Test labels are not used for selection. The finalized panel uses only the
approved backbone comparisons recorded in the Fig. 2 config.

External-model benchmark code is under `external_models/`. Exact training tables,
external baseline weights, and their selected fine-tuned checkpoints are not in Git;
the seven included native InvaRNA checkpoints are a separate inventory.
See `benchmarks/configs/external_llm_te.yaml` and `external_models/README.md` for
the input and download records.

The saved selected-checkpoint predictions are retained in a private benchmark
archive, not in Git. Only where those prediction tables are supplied,
the reported test R2 values can be recalculated without downloading models:

```bash
python benchmarks/heldout_human_te/verify_external_models.py \
  --output outputs/external_llm_test_r2.csv
```

Full external-model checkpoint inference additionally requires the upstream weights
and model-specific dependencies. After obtaining them in the matching environment,
run for example:

```bash
python external_models/inference/evaluate_te_checkpoint.py \
  --model evo2_8k \
  --output outputs/evo2_8k_human_test.csv
```

The common configuration records each sequence region, maximum context, selected
checkpoint, and expected result. The current 2026-08-05 `nm.pdf` uses `evo2_8k`.

The 1,115-row human WT validation and test splits themselves are included in the
GitHub checkout at `assets/benchmark_data/human_te/human_val_wt.parquet` and
`assets/benchmark_data/human_te/human_test_wt.parquet`. They contain the sequence,
frame, measured TE, and historical teacher-derived columns. The external-model
prediction tables and larger training inputs are not distributed in this release.

## Fig. 3b cross-species benchmark

The current paper shows five RPFdb species for InvaRNA, RiboNN, and UTR-LM. The
displayed Fig. 3B summary and merged plotting table are in
`assets/manuscript_figures/full/fig3_current_nm/`, so the main Fig. 3 redraw works
from Git. The three separate `F3B_pred_*.csv` files required to recompute all 15
Spearman correlations are not in Git. After supplying those historical files in
that directory, run:

```bash
python figures/fig3/fig3b_multispecies.py
```

This entry intentionally reads the three prediction tables separately. The historical
RiboNN table has 17,997 rows while the other two have 18,000; using only their merged
intersection would slightly perturb the displayed model-specific correlations.

## Human/mouse WT species ablation

The cross-species control compares human WT training, human plus RBH-filtered mouse
WT training, and human plus all mouse WT training. Each condition has exactly three
declared seeds (`22222`, `3407`, and `9713`); every checkpoint is selected using only
the human validation R² and evaluated on the human-only test split.

The nine training YAML files under `sft/configs/species_ablation/` are in Git. The
exact input tables, saved epoch metrics, and selected prediction files under
`assets/benchmark_data/rbh_sensitivity/full/`, the nine runtime checkpoints under
`assets/checkpoints/ablations/rbh_sensitivity/`, and the historical RBH reference
table are not. They remain in the author's private archive. Only after restoring
those inputs can the full audit run:

```bash
python benchmarks/rbh_sensitivity/audit.py
```

This recomputes all nine validation/test R² values, paired tests, training/split
composition, RBH exclusion, best-validation epoch selection, checkpoint hashes, and
legacy-checkpoint loading. It also validates per-run and summary tables under
`assets/manuscript_figures/results/`; those tables are not in Git either.

## Fig. 3g external MPRA

The current `nm.pdf` Fig. 3g bar (unchanged from July 17) compares UTR-LM, RiboNN, and InvaRNA on ten Optimus
5-Prime MPRA libraries. The historical calculation uses 3,000 rows per library,
Spearman correlation within each library, then the arithmetic mean and sample
standard deviation (`ddof=1`) across the ten correlations. The exact parameter and
checksum record is `benchmarks/configs/fig3_panel_g.yaml`.

The main Fig. 3 redraw uses the committed
`assets/manuscript_figures/full/fig3_current_nm/F3G_external_MPRA_spearman_by_sample.csv`.
The 3,000-row-per-library panel input, 100,000-row-per-library source input, and model-specific
prediction files named in `benchmarks/configs/fig3_panel_g.yaml` are not in Git.
To recalculate the historical ten-library correlations from those private inputs
after supplying them, run:

```bash
python figures/fig3/fig3g_mpra.py
```

This checks all source hashes and writes per-library metrics, the three-bar summary,
PNG, and PDF under `results/fig3_panel_g/`. Expected mean +/- SD values are UTR-LM
0.085538806 +/- 0.119267787, RiboNN 0.259656219 +/- 0.094617926, and InvaRNA
0.353215975 +/- 0.093279511.

The canonical RiboNN value was computed from the preserved 100,000-row-per-library
submission prediction file using the original sample-then-drop-missing order. The
separate 30,000-row checkpoint rerun is retained privately as runtime evidence and
gives 0.259762102 +/- 0.094540644; it is not the manuscript source. The May 22
combined figure under `figure_data/fig3/combined_20260522/` is historical-only
and is not in Git.

Checkpoint inference on the exact 30,000 sequences requires the omitted MPRA
input table, the relevant external weights, and their model-specific dependencies.
After supplying these, the rerun commands are:

```bash
python benchmarks/mpra/predict.py --model invarna \
  --output results/fig3_panel_g/pred_invarna_rerun.csv
python benchmarks/mpra/predict.py --model ribonn \
  --output results/fig3_panel_g/pred_ribonn_rerun.csv
python benchmarks/mpra/predict.py --model utrlm \
  --output results/fig3_panel_g/pred_utrlm_rerun.csv

python figures/fig3/fig3g_mpra.py --skip-hash-check \
  --expected-tolerance 5e-4 \
  --invarna-predictions results/fig3_panel_g/pred_invarna_rerun.csv \
  --ribonn-predictions results/fig3_panel_g/pred_ribonn_rerun.csv \
  --utrlm-predictions results/fig3_panel_g/pred_utrlm_rerun.csv
```

The RiboNN production wrapper calls `runs.csv.head(1)` and `n_folds=1`, resolving to
run `057db008d02743f4b319c5961b00ed41`. UTR-LM averages ten HEK checkpoints and uses
the last 100 nt of each 5' UTR. The exact preserved tables, rather than a fresh GPU
rerun, are the byte-stable source for the submitted panel because low-order floating-
point differences can occur across CUDA runtimes.

```bash
python benchmarks/evaluate.py --input PREDICTIONS.csv \
  --truth te_ratio --prediction pred_InvaRNA0412 --group species \
  --truth-transform log10 --output METRICS.csv
```
