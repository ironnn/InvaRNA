# Model-integrity inventory

This inventory separates model completeness from data completeness. A model is marked
complete only when its architecture/loading code, exact checkpoint, provenance role,
and immutable checksum are all known. Local-only and third-party files are not made
GitHub-safe merely by being present on disk.

## InvaRNA-native models

| Group | Required artifacts | Local status | Integrity record | Publication action |
|---|---:|---|---|---|
| Mamba–Motif-MoE backbone | 1 weight file + architecture/config/tokenizer | Complete | `assets/checkpoints/REQUIRED_SHA256SUMS` | Publish the weight in the seven-file asset package |
| Privileged TE teacher | 1 LightGBM model + 5 metadata files + exact source snapshot | Complete | `assets/checkpoints/REQUIRED_SHA256SUMS`; teacher audit | Publish the model only; historical teacher inputs are not distributed |
| Fig. 3 selected ablations | 3 Stage-2 checkpoints | Complete locally | `assets/checkpoints/ABLATION_SHA256SUMS` | Private provenance only; frozen figure tables need no weights |
| Human/mouse WT species ablation | 3 conditions x 3 seeds = 9 Stage-2 checkpoints | Complete locally | `assets/checkpoints/SPECIES_ABLATION_SHA256SUMS`; species-ablation audit | Private provenance only; saved prediction tables suffice for the displayed values |
| Native inference registry | 5 Stage-1 + 14 Stage-2 TE + 2 half-life checkpoints | Complete locally | `assets/checkpoints/INFERENCE_SHA256SUMS` | Private historical inventory; only four TE and one HL weights are distributed |
| Fig. 5 updated design predictors | Four TE ensemble members and `hl_new` are contained in the native registry | Complete locally; prospective configuration | `assets/checkpoints/REQUIRED_SHA256SUMS` | Publish in the minimal native-checkpoint package |

Exact native inventory:

- Backbone: `model_weights0718step13500.pt`.
- Privileged teacher: `teacher_model.pkl`.
- Stage-1 registry: `evo`, `hlevo_e2`, `hlevo_e7`, `noevo_s1`, `test_s1`.
- Stage-2 TE registry: `InvaRNA0412`, `noevo`, `notaylor`, `old1`, `old2`,
  `only_taylor_beidian`, `taylor_dist`, `taylor_dist_e14`, `taylor_dist_e21`,
  `w0`, `w1`, `w2`, `world_old`, `wt_human`.
- Half-life registry: `hl0704`, `hl_new`.
- Selected Fig. 3 ablations: `ablation_1k_org_ep22`,
  `ablation_2r_org_ep28`, `ablation_3kl_tay_ep46`.
- Human/mouse WT species ablation: `human`, `human_mouse_rbh_filtered`, and
  `human_mouse_full`, each for seeds `22222`, `3407`, and `9713`.

The current Fig. 5 design configs use four TE predictors: `w0`, `w1`, `w2`, and
`final_tdc`, plus the separate half-life model `hl_new`. The first three are renamed
distribution files for the historical `wt`, `world_new`, and `world_new07522`
weights, respectively; the serialized bytes and SHA-256 values are unchanged.
This four-TE ensemble is a prospective configuration update, not evidence that the
historical manuscript Fig. 5 runs used four TE models. The exact historical
all-task production launcher remains unrecovered.

The three selected Fig. 3 ablations map to the authenticated runs as follows:

| Checkpoint | Experimental condition | Held-out-human R² |
|---|---|---:|
| `ablation_2r_org_ep28.ckpt` | matched-random, absolute teacher targets | 0.7292183 |
| `ablation_1k_org_ep22.ckpt` | K80, absolute teacher targets | 0.7346692 |
| `ablation_3kl_tay_ep46.ckpt` | K80, WT-anchored distance-scaled TDC | 0.7393672 |

`InvaRNA0412.ckpt` and `taylor_dist_e14.ckpt` have different serialized-file hashes
but identical 357-key state dictionaries and identical held-out predictions in the
recorded matched-runtime test. They remain separate provenance artifacts.

The WT species ablation uses human-only validation and test splits for every condition.
Each checkpoint is the maximum-validation-R² epoch from its 50-epoch run. Recomputed
held-out-human test R² (mean +/- sample SD across the three declared seeds) is:

| Training data | n | Test R² |
|---|---:|---:|
| Human WT | 3 | 0.653592 +/- 0.003854 |
| Human WT + RBH-filtered mouse WT | 3 | 0.675682 +/- 0.001650 |
| Human WT + all mouse WT | 3 | 0.712713 +/- 0.004549 |

The exact per-run values, selected epochs, inputs, configs, original metrics, saved
predictions, and byte-identical checkpoint copies are validated by
`benchmarks/rbh_sensitivity/audit.py`. Historical multi-GPU prediction files
contain 1,116 rows for the 1,115-row human split because the distributed sampler pads
to world size; the audit preserves the reported calculation and does not silently
replace it with a post hoc de-duplicated metric.

## Privileged-teacher dependency closure (private historical inputs)

The standalone workflow does contain the previously questioned dependencies:

| Dependency | Repository location | Status |
|---|---|---|
| Engineered-feature extractor | `data_generation/label_generation/teacher/source/utils/lgbm_feature_extract_from_str.py` | Included source |
| Human half-life table | `assets/training_data/half_life/teacher_reference/human_time.csv` | Private; not distributed |
| Mouse half-life table | `assets/training_data/half_life/teacher_reference/mouse_time.csv` | Private; not distributed |
| Human→mouse RBH used by student-label leakage filtering | `assets/training_data/half_life/teacher_reference/rbh_human_to_mouse.parquet` | Private; not distributed |
| Mouse→human companion RBH | `assets/training_data/half_life/teacher_reference/rbh_mouse_to_human.parquet` | Private; not distributed |
| Prepared human/mouse teacher tables | `assets/training_data/teacher/full_inputs/` | Private; not distributed |
| Frozen feature order and split indices | `assets/checkpoints/teacher/metadata/` | Included |
| Production training/scoring scripts | `data_generation/label_generation/teacher/source/` | Included |

The deployed model has exactly 3,274 named inputs: 3,072 `feat_*` columns, 201
engineered sequence columns, and `hl`. RBH is not a fitted teacher feature; it is used
by the later orthology-aware leakage filter. Missing fitted features must raise an
error; they must never be imputed as zeros.

The preserved extractor lists `Struct` in `FEATURE_LIST`, but its string-based `fe()`
path adds no structural columns, and the frozen model contains no `Struct` feature
names. The 201-column count above is therefore the actual fitted contract, not a
reconstructed interpretation of the comment in the legacy extractor.

The production half-life merge maps version-stripped Ensembl gene IDs to
`half-life (PC1)` and falls back to each prepared row's `pred_score` when no mapping is
available. In the frozen inputs, 10,879/11,147 human rows and 10,950/11,433 mouse rows
map to the supplied half-life tables; 268 human and 483 mouse rows use that recorded
fallback. This behavior is preserved rather than replaced with imputation.

Two method/provenance observations are intentionally not “fixed” in the review copy:

1. The production teacher training script uses all 11,433 mouse rows. After the exact
   `_mut0` suffix handling used by the later homology filter, 1,854 of the 2,230 frozen
   human validation/test transcripts have RBH partners and 1,772 partner rows occur in
   the mouse teacher-training table. Human validation/test rows themselves are not in
   training, so this is not direct human-sample leakage. It is a cross-species training-
   set design factor assessed by the separate human / RBH-filtered human+mouse / full
   human+mouse ablation. Orthology-aware removal was applied later to student-label
   tables, not to the already-fitted teacher.
2. All five teacher candidates were evaluated on the human test split and serialized
   with test R² in their filenames. Trial 2 is the deployed model; the preserved source
   optimizes validation R², but the surviving notes also call Trial 2 the highest-test-
   R² model. Whether test performance affected deployment selection needs author
   confirmation.

## External benchmark models

Code for CodonBERT, DNABERT-2, Evo2, LucaOne, mRNABERT, Orthrus, RiboNN, RNA-FM, and
UTR-LM is under `external_models/`. External weights are deliberately not
included in the publication asset archive. Their source, version/revision, license,
and per-file SHA-256 records are in `external_models/DOWNLOADS.tsv` and
`external_models/CHECKSUMS.tsv`. `benchmarks/configs/external_llm_te.yaml` records
the selected checkpoint names and context lengths. The current 2026-08-05 `nm.pdf`
uses Evo2-8k.

Presence is not redistribution permission. The external model files are therefore
download-on-demand only, and restricted CodonBERT/RiboNN artifacts are not part of
the public asset deposit. The saved external-model prediction tables are sufficient
for the manuscript figures.

Selected fine-tuned runs are: `codonbert`, `dnabert2`, `evo2_8k`, `lucaone`,
`mrnabert`, `orthrus`, `ribonn`, `rnafm`, `utrlm`,
and the `invarna_human_mouse` common-split control. These names identify the records
in `external_models/DOWNLOADS.tsv`; they do not imply that weights are redistributed.

## Verification

```bash
(cd assets/checkpoints && sha256sum -c REQUIRED_SHA256SUMS)
python tests/verify_assets.py
```

Teacher-source and external-baseline audits require historical full input or
prediction tables that are not distributed in this release. The teacher audit
can be run only where the private inputs are available; it is not a check of
the seven-file weight package.

For the private full historical model inventory, verify `SHA256SUMS`,
`INFERENCE_SHA256SUMS`, `ABLATION_SHA256SUMS`, and
`SPECIES_ABLATION_SHA256SUMS` separately and run
`python benchmarks/rbh_sensitivity/audit.py` with its full benchmark inputs.
Those four historical checksum records cover all 35 native model artifacts: one backbone, one
teacher, 21 inference-registry checkpoints, three selected Fig. 3 ablations, and nine
human/mouse WT species-ablation checkpoints.
