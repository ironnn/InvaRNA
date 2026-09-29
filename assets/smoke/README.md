# Real-data smoke inputs

This directory contains only tiny deterministic excerpts of the preserved production
inputs. They are copied from real rows; sequences, labels, fitted teacher features,
split membership, and benchmark values are not fabricated. The excerpts are intended
for schema, loader, inference, and short integration checks. They are not statistically
representative and must not be used to reproduce manuscript metrics or train a useful
model.

The complete training and evaluation datasets are deliberately excluded from GitHub.
They remain in Git-ignored local or external archives. `MANIFEST.tsv` records the local
source and deterministic selection rule for every excerpt.

Coverage:

- `pretrain/`: two complete records from each production MLM FASTA split;
- `student/`: three real rows from each human/mouse WT train/validation/test table;
- `synthetic/`: a WT anchor and real K80, matched-random, and final-TDC rows; raw FM
  pool excerpts; and three `mut201`–`mut215` FM training rows with absolute teacher
  labels;
- `teacher/`: complete 3,274-feature teacher rows plus small half-life/RBH excerpts;
- `evaluation/`: one row per MPRA library and RPFdb species, evaluation-only;
- `design/`: three rows from the recovered NGF PPO trajectory.

Rebuild from the local full datasets with:

```bash
python data_generation/utilities/build_smoke_data.py
```

The full datasets are required only to rebuild these excerpts. Once generated, the
repository smoke test validates their hashes and schemas without accessing the full
data.
