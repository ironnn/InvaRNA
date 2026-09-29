# InvaRNA

Code and frozen computational assets accompanying the anonymous manuscript
“InvaRNA enables transferable mRNA sequence–function modeling from sparse functional
measurements.”

For manuscript verification, start from [`figures/`](figures/README.md):

```bash
bash figures/fig2/reproduce.sh
bash figures/fig3/reproduce.sh
bash figures/fig4/reproduce.sh
bash figures/fig5/reproduce.sh
```

## Repository structure

| Path | Purpose |
|---|---|
| `assets/` | Minimal frozen figure tables, sequences, smoke fixtures, and provenance |
| `data_generation/` | K80, matched-random, Flow-Matching-pool consumption, teacher scoring, and TDC label construction |
| `backbone/` | 274-species preprocessing and Mamba–Motif-MoE pretraining |
| `sft/` | Sequence-only student supervised training |
| `inference/` | Frozen TE/half-life inference, embeddings, and routing extraction |
| `benchmarks/` | Held-out human, RPFdb, MPRA, representation-transfer, RBH, and ablation evaluations |
| `external_models/` | External baseline implementations and license evidence |
| `figures/` | Fig. 2–5 and Extended Data orchestration, analysis, and plotting |
| `tests/` | Smoke, split-integrity, checkpoint, and manuscript-value checks |
| `src/invarna/` | Import-compatible internal package used by the public entry points and released checkpoints |
| `docs/` | Reproducibility, provenance, figure audit, licensing, and known limitations |

The `src/invarna/` package name is intentionally retained because released PyTorch
checkpoints and compatibility shims depend on its module paths. Reviewer-facing tasks
should use the top-level directories above.

All repository-owned commands, configurations, data references, outputs, and copied
baseline runtime defaults use paths relative to the repository root. The workflow does
not depend on the original development checkout or server filesystem. Run commands from
the repository root; `tests/test_relative_paths.py` enforces this portability rule.

## Quick start

```bash
conda env create -f environment.yml
conda activate mamballm
python -m pip install -e .
```

The figure-only native asset archive is distributed separately from GitHub. It contains
only the frozen tables needed by the Fig. 2--5 plotting entry points plus small smoke,
sequence, and provenance files; training corpora, benchmark archives, raw design
trajectories, and all model weights are intentionally excluded. External baseline
weights are also excluded; their download metadata and checksums are under
`external_models/`. Extract or mount the native archive as `assets/`, then verify it:

```bash
python tests/verify_assets.py
python external_models/audit_licenses.py
python tests/smoke/test_pipeline.py --core-only
```

The full checkpoint/inference/routing smoke requires the separately mounted private
checkpoint archive and CUDA/Triton kernels on an NVIDIA GPU:

```bash
python tests/smoke/test_pipeline.py --device cuda:0
```

`assets/smoke/` contains only small rows copied from the real inputs. Smoke data check
schemas and code paths; they do not reproduce manuscript statistics.

The GitHub checkout is standalone for code inspection, smoke fixtures, label logic,
and the lightweight RL smoke test. The published figure redraws are standalone after
the compact `assets/` directory is placed at the repository root; full model inference
and training require the separately maintained private archives.

## Inference

Input CSV/parquet tables require the sequence and frame columns documented in
[`docs/INFERENCE.md`](docs/INFERENCE.md).

```bash
python inference/predict_te.py \
  --model final_tdc --input INPUT.parquet --output te_predictions.csv

python inference/predict_half_life.py \
  --input INPUT.parquet --output half_life_predictions.csv

python inference/extract_embeddings.py \
  --input INPUT.parquet --output embeddings.pkl --gpus 0
```

The canonical deployed TE checkpoint is
`assets/checkpoints/te_student/final_tdc.ckpt`. Historical filenames and hashes are
restricted to model-provenance records and the non-canonical provenance archive.

## Computational workflow

The repository preserves the manuscript workflow without changing algorithms or
parameters:

1. Ensembl mature-mRNA preprocessing and fixed CDS@1000 framing.
2. 274-species Mamba–Motif-MoE masked-language-model pretraining.
3. Frozen embedding extraction and privileged-teacher feature construction.
4. K80-derived local variants, which may contain multiple substitutions.
5. Non-local Flow-Matching UTR replacements and matched-random controls.
6. WT-anchored TDC plus distance scaling for K80 variants only; Flow-Matching variants use absolute teacher predictions without WT anchoring or distance scaling.
7. Sequence-only student training with gene-disjoint human train/validation/test splits.
8. Held-out human, RPFdb, MPRA, representation-transfer, routing, dORF, and design analyses.

Synthetic variants originate only from training anchors. Validation/test/external
labels do not enter teacher fitting, synthetic-label construction, or student
training. RPFdb and MPRA are external evaluation only. Matched-random training uses
absolute teacher predictions; TDC shifts are post hoc QC only.

## Training entry points

```bash
# Backbone pretraining from scratch (expensive)
python backbone/pretrain.py --config backbone/configs/mamba_motif_moe.yaml

# Privileged teacher
python data_generation/label_generation/teacher/train_teacher.py \
  --config data_generation/label_generation/configs/te_teacher.yaml

# Synthetic variants
INVARNA_GENERATION_CONFIG=data_generation/evolutionary_variants/configs/k80.yaml \
  python data_generation/evolutionary_variants/generate_k80.py
python data_generation/evolutionary_variants/generate_matched_random.py --help
python data_generation/flow_matching/generate.py

# Final sequence-only student
python sft/train.py --student-config sft/configs/tdc_final.yaml
```

Pretrained checkpoints and frozen intermediates are the default reviewer route; no
expensive training must be rerun to verify reported values.

## Manuscript result map

| Result | Main entry | Frozen inputs |
|---|---|---|
| Fig. 2 | `figures/fig2/reproduce.sh` | `assets/manuscript_figures/fig2_compact/` |
| Fig. 3 | `figures/fig3/reproduce.sh` | `assets/manuscript_figures/full/fig3_current_nm/` |
| Fig. 4 | `figures/fig4/reproduce.sh` | `assets/manuscript_figures/fig4_compact/` |
| Fig. 5 | `figures/fig5/reproduce.sh` | selected tables under `assets/manuscript_figures/full/fig5/` |
| Extended Data | `figures/extended_data/ext_fig*/reproduce.py` | explicit availability checks; final panel tables remain unavailable |

Panel-level closure and unresolved raw-data provenance are recorded in
[`docs/FIGURE_AUDIT.csv`](docs/FIGURE_AUDIT.csv) and
[`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md).

## License

The top-level MIT license applies only to InvaRNA-authored material. Third-party code,
model artifacts, and checkpoints retain their original licenses; see
[`docs/THIRD_PARTY_LICENSES.md`](docs/THIRD_PARTY_LICENSES.md) and
[`external_models/THIRD_PARTY_LICENSES.tsv`](external_models/THIRD_PARTY_LICENSES.tsv).
CodonBERT and RiboNN remain academic/non-commercial components, so an unrestricted
public bundle must exclude them or obtain explicit redistribution permission.
