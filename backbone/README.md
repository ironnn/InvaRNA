# Backbone

This directory is the reviewer-facing entry for 274-species preprocessing and
self-supervised Mamba–Motif-MoE pretraining. The import-compatible implementation is
under `src/invarna/`; these modules expose it without duplicating the algorithm.

```bash
python backbone/pretrain.py --config backbone/configs/mamba_motif_moe.yaml \
  --train_file assets/manuscript_figures/train0606.fasta \
  --val_file assets/manuscript_figures/val0606.fasta
```

For a short GPU smoke run, use the checked-in toy excerpts:

```bash
python backbone/pretrain.py \
  --train_file assets/training_data/pretraining/toy/toy_train.fasta \
  --val_file assets/training_data/pretraining/toy/toy_val.fasta \
  --devices 1 --max_steps 2 --batch_size 2 --val_check_interval 2 \
  --limit_val_batches 1.0 --ckpt_every_n_steps 1 --num_workers 0
```

The released checkpoint is `assets/checkpoints/backbone/pretrained_step13500.pt`.
