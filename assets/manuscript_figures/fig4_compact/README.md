# Fig. 4 computational source data

These files are direct copies of the recovered final-figure inputs and are tracked
in the GitHub repository. Their hashes are recorded in `SHA256SUMS`.

- `_f4a_local_s0.csv`, `_f4a_local_s1.csv`: 5,000 elements × four experts; local
  element shuffle, 20 permutations, InvaRNA0412.
- `fig4a_statistics_iqr.csv`: statistics recomputed from the two local files using
  the final 1.5×IQR rule.
- `dorf_orfs.csv`: the gene-symbol list used by the strict audit.
- `dorf_moe_summary_InvaRNA0412.csv`: compact case-study values used by the strict
  audit. The large background permutation table and transcript annotations are not
  needed by the current redraw and are excluded.
- `dorf_zscore_genome_wide_InvaRNA0412.csv`: 8,241 genes, 50 whole-3'UTR
  permutations per gene.

No wet-lab outcome was used to compute the null distributions or genome-wide rank.
