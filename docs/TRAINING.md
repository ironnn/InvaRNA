# Training

Backbone MLM pretraining uses `invarna.training.pretrain`; student TE fine-tuning uses
`invarna.training.train_student`; the LightGBM teacher entry point is
`invarna.training.train_teacher`. Experiment defaults live under `configs/`.

```bash
python -m invarna.training.pretrain --help
python -m invarna.training.train_student --help
python -m invarna.training.train_teacher --help
```

The released backbone uses a 10,000-nt frame, A/C/G/T-only MLM masking, and the
architecture in `backbone/configs/mamba_motif_moe.yaml`. The overlength-5'UTR
handling difference between the selected Stage-2 trainer and the embedding/later
benchmark implementation is retained as a documented implementation limitation.

The original 274-species pretraining FASTA, exact species manifest, and historical
Ensembl preprocessing script are not provided. From-scratch users must download
release-111 data directly from Ensembl and supply their own train/validation FASTA
files. To reproduce downstream analyses without rebuilding the corpus, use the
released step-13,500 backbone checkpoint.
