# Data and synthetic augmentation

K80 and flow-matching augmentation entry points live in `data_generation/`, with the
shared implementation in `src/invarna/synthetic/`. Configuration paths are relative
to the repository root.

```bash
bash data_generation/reproduce_training_data.sh
```

Required WT pickle tables contain `transcript_id`, `gene_id`, `mrna`, `utr5_size`,
`cds_size`, `utr3_size`, and `mean_te`. Place reviewed inputs under
`assets/training_data/synthetic/generation_inputs/`. Newly generated artifacts are
written to `assets/training_data/synthetic/generated/` and are ignored by Git.

The complete FM pools remain local under `assets/training_data/synthetic/flow_matching/` and are ignored by
Git. Small real-row excerpts are in `assets/smoke/synthetic/`. Matched random controls
are available through `invarna.synthetic.matched_random.mutate_matched`.

## GitHub data policy

No complete training dataset is uploaded to GitHub. The repository tracks only the
small excerpts under `assets/smoke/`; each is deterministically copied from the real
preserved source by `data_generation/utilities/build_smoke_data.py`. Full inputs remain in ignored
local/external storage. Smoke excerpts preserve real values and schemas but are not
representative and cannot reproduce manuscript statistics.

## Backbone pretraining corpus

The historical 274-species pretraining FASTA is not included. The exact species
manifest and Ensembl-to-mature-mRNA construction script were not preserved and are
not supplied by this repository. For a new from-scratch pretraining run, download the
relevant sequence and annotation files directly from Ensembl release 111 and prepare
train and validation FASTA files for the pretraining CLI. The resulting corpus is a
new user-prepared corpus, not an exact reconstruction of the 10,611,071-transcript
corpus used to train the released backbone.
