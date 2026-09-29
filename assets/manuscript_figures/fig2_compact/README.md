# Figure 2 computational source data

These compact tables are direct copies of the finalized Figure 2 plotting data and
are installed from the separate publication figure-data archive; they are not GitHub
blobs. `SHA256SUMS` (tracked in Git) records their byte identity. The primary entry
point after archive installation is:

```bash
python figures/fig2/reproduce.py
```

That command checks every hash and reported value, then writes the combined PDF and
PNG under `results/fig2/`. Use `--check-only` for the numerical consistency check.

## Panels

- `F2A_left_species_UMAP.csv`: 7,000 displayed points, exactly 1,000 transcripts
  from each of seven species. Parameters were `n_neighbors=30`, `min_dist=0.3`, and
  `random_state=42`. To avoid UMAP/Numba version drift, the committed coordinates are
  those recovered at vector precision from the submitted panel. The source embeddings
  remain in the separately distributed asset archive.
- `F2A_right_centroid_vs_divergence.csv`: all 21 species pairs, using Euclidean
  distances between embedding centroids and the exact TimeTree values used by the
  finalized script. Spearman rho is 0.7584483932 (`p=6.7614253e-05`).
- `F2B_*.csv`: frozen-embedding LightGBM test results for mean ribosome load,
  mRNA half-life, and expression level. The full benchmark protocol is in
  `supplementary_table_backbone_transfer_benchmark_protocol.csv`; exact parameters
  and selection rules are in `benchmarks/configs/fig2.yaml`.
- `F2C_left_routing_curve.csv` and `F2C_right_region_specialization.csv`: routing
  probabilities from the released step-13,500 backbone on the archived seed-42 batch.
  The left curve uses a centered 80-position rolling mean.
- `F2C_sample_manifest_seed42.csv`: the 24-row batch order used to validate the
  frozen routing tables. The aligned sequences and masked token IDs are deliberately
  excluded from the GitHub plotting bundle; they are only needed for an optional
  checkpoint replay, not for the published redraw.

## Finalized F2B display decisions

The finalized panel contains only the approved model comparisons. The compact archive
does not include unapproved baseline rows or their model artifacts.

CodonBERT was omitted only from mean ribosome load because that benchmark contains
bare 5-prime-UTR fragments and no CDS for a codon representation. The old MPRA README
value for InvaRNA (`R2=0.286`) is superseded by the finalized source table value
(`R2=0.375840`, Spearman `0.566700`).

No training or experimental-screening feedback is used by these plotting tables.
