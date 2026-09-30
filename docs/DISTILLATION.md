# Distillation contract

The label rule is source-specific and enforced by `invarna.distillation.tdc`:

- WT: measured TE.
- K80 local variants: measured WT + teacher mutant–WT delta × normalized Hamming distance; alpha 1.0.
- Flow-Matching non-local UTR replacements: absolute teacher prediction, no anchor/scaling.
- Matched-random control: absolute teacher prediction during training. Matched-random TDC shifts are post hoc Extended Data Fig. 3 QC only.

At least one validation/test exclusion table is mandatory. The command fails if any
held-out transcript family appears among label-construction anchors.

## Privileged teacher

The production teacher source snapshot is preserved under
`data_generation/label_generation/teacher/source/`. In particular, this includes the
exact engineered-feature extractor from the standalone workflow, the production
five-candidate LightGBM training script, K80/matched feature extraction and scoring,
and the later orthology-aware student-label filter.

The fitted model uses 3,274 ordered inputs: 3,072 backbone embedding columns, 201
engineered sequence columns, and one half-life column. Its frozen feature order, split
indices, trial record, and hyperparameters are under `assets/checkpoints/teacher/metadata/`.
The historical human/mouse half-life tables, bidirectional RBH tables, and prepared
teacher feature tables are retained in a private, non-release archive. They are
not part of the public seven-weight package. Scoring an already engineered feature
table with the released teacher model does not require those historical tables;
retraining or auditing the original teacher-input construction does.

With the private historical inputs installed, run the integrity and method audit with:

```bash
python data_generation/label_generation/teacher/audit.py
```

The audit reports that the teacher used all mouse rows before the later RBH exclusion
was applied to student-label tables. Human validation/test rows themselves are absent
from training; mouse-ortholog inclusion is treated as a cross-species training-set
ablation factor, not direct human-sample leakage. The audit does not retrain or replace
the deployed checkpoint. See `docs/MODEL_PROVENANCE.md` for the exact counts.
