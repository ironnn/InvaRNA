# Privileged-teacher source snapshot

This directory preserves the experiment code used to build and apply the fitted
LightGBM privileged teacher. It is retained as a source snapshot rather than rewritten
behind the publication-facing wrappers.

The files have the following roles:

- `utils/lgbm_feature_extract_from_str.py`: exact engineered sequence-feature
  extractor used by the standalone ablation workflow.
- `teacher_train.py`: production five-candidate LightGBM training script. It creates
  the human 80/10/10 split with seed 111 and adds the complete mouse table to training.
- `teacher_verify.py` and `check_teacher.py`: production model checks.
- `pipeline_k80/` and `pipeline_matched/`: feature extraction and teacher-scoring
  stages used by the synthetic-label workflows.
- `label_engineering_homosplit_v2.py`: later student-label filtering code that removes
  held-out human transcripts and their reciprocal-best-hit mouse orthologs.
- `teacher_allmut.py`: production combined scoring workflow.

Required local-only inputs are stored outside ordinary Git:

```text
assets/training_data/teacher/
├── reference/
│   ├── human_time.csv
│   ├── mouse_time.csv
│   ├── rbh_human_to_mouse.parquet
│   └── rbh_mouse_to_human.parquet
└── training/
    ├── train_ready_human_regional_strict_20260117_pred_8gpu_bf16.pkl
    └── train_ready_mouse_regional_strict_20260117_pred_8gpu_bf16.pkl
```

The deployed model and frozen metadata are under `assets/checkpoints/teacher/`. The model
contains 3,274 fitted inputs: 3,072 backbone embedding values, 201 engineered sequence
features, and one half-life feature. RBH is used by the later student-label leakage
filter, not as a fitted teacher feature. The human→mouse RBH table is from the TE
standalone workflow; the reverse companion table is from the separate half-life
standalone snapshot. `mrl_isoform_resolved.npz` is an Orthrus benchmark asset and is
not an input to this teacher.

The historical scripts use paths relative to their original directories. They are
preserved for provenance and are not silently redirected. See `docs/DISTILLATION.md`
and `data_generation/label_generation/teacher/audit.py` for the publication-tree mapping.

Important: the production teacher was fit before the later orthology-aware student-
label filtering step. The frozen human validation/test transcripts have 1,854 RBH
matches, and 1,772 corresponding mouse rows occur in the teacher training table. The
human validation/test rows themselves are not in training. The repository therefore
records this as a cross-species training-set design choice, not direct human-sample
leakage, and does not substitute a retrained model.
