# Interpretability

`invarna.interpretability` contains routing aggregation, feature-shuffle controls, and
InvaRNA-native scoring helpers for dORF and yeast analyses. Publication source tables remain
pending provenance review.
# Fig. 4 element routing and dORF nomination

The final Fig. 4A computation used `InvaRNA0412` and four `gate_fwd` experts. For
each annotated element, only the element interval was shuffled, preserving local
nucleotide composition. The native sequence and 20 shuffles were passed through
the model; expert softmax probabilities were averaged over the same interval and
converted to

`Z = (native mean - permutation mean) / (permutation population SD + 1e-9)`.

The five plotted conditions were strong uORF, RBP in 5'UTR, CDS or 3'UTR, and miRNA
in 3'UTR. At most 1,000 elements per condition were sampled with seed 42. RBP and
miRNA coordinates in the parsed source were 1-based inclusive and were converted
to zero-based half-open intervals. uORF coordinates were reconstructed as
`end = utr5_size - uORFCDSdistance`, `start = end - uORFlength`. Final aggregation
removed 1.5×IQR outliers within each condition/expert group and used a two-sided
one-sample t-test against zero.

Run a prepared unified element table with:

```bash
python figures/fig4/routing_analysis/element_routing.py \
  --input assets/processed/fig4_elements.parquet \
  --output results/fig4a_element_zscores_s0.csv --num-shards 2 --shard-index 0
python figures/fig4/routing_analysis/element_routing.py \
  --input assets/processed/fig4_elements.parquet \
  --output results/fig4a_element_zscores_s1.csv --num-shards 2 --shard-index 1
```

The dORF nomination workflow was:

1. Extract each mature transcript's 3'UTR using `utr5_size + cds_size`.
2. Call `orffinder.getORFs(minimum_length=75, remove_nested=True)`.
3. Calculate 3'UTR RPKM from RPFdb featureCounts as CPM divided by feature kb.
4. Use the median across nonzero samples and retain ORF-positive transcripts with
   median 3'UTR RPKM > 1.
5. Expand retained transcripts to individual ORF rows. The recovered `orf_rpkm`
   field scales the transcript's 3'UTR RPKM by `utr3_length / orf_length`; it is an
   assumed uniform-density proxy, not ORF-specific read counting.
6. Choose the highest-`orf_rpkm` transcript/ORF row per gene symbol for the
   genome-wide routing calculation.
7. Insert its native 3'UTR behind the fixed reporter 5'UTR/CDS backbone, shuffle the
   entire 3'UTR 50 times (seed 99), compute four whole-3'UTR expert Z-scores, and
   rank genes by descending Ribosome-expert Z-score. Case-study distributions used
   1,000 permutations.

Commands:

```bash
python figures/fig4/candidate_prioritization/nominate_dorf.py \
  --transcripts assets/processed/human_transcripts.parquet \
  --output results/orf_annotations.csv \
  --counts-matrix assets/processed/rpfdb_3utr_counts.csv \
  --gene-lengths assets/processed/rpfdb_3utr_lengths.csv \
  --candidates-output results/dorf_candidates.csv \
  --orfs-output results/dorf_orfs.csv

python figures/fig4/candidate_prioritization/dorf_routing_zscore.py \
  --dorf-orfs assets/manuscript_figures/fig4_compact/dorf_orfs.csv \
  --transcripts assets/processed/human_transcripts.parquet \
  --output results/dorf_zscores_s0.csv --num-shards 2 --shard-index 0
python figures/fig4/candidate_prioritization/dorf_routing_zscore.py \
  --dorf-orfs assets/manuscript_figures/fig4_compact/dorf_orfs.csv \
  --transcripts assets/processed/human_transcripts.parquet \
  --output results/dorf_zscores_s1.csv --num-shards 2 --shard-index 1

python figures/fig4/audit.py --strict
```

The large mature-transcript table, raw RPFdb featureCounts files, and licensed motif
annotation inputs are not bundled. The compact recovered outputs are included for
review without rerunning the GPU-intensive permutations.

The production InvaRNA0412 rerun split both calculations into two contiguous GPU
chunks. F4A restarted seed 42 in both processes; genome-wide dORF used seeds
`99 + shard_index`. The shard options above preserve that behavior. Merge the two
output CSVs before ranking; the bundled merged output is the recovered reference.

## Provenance

Core logic was copied from the recovered production files, with only paths, device
selection, and CLI I/O parameterized for anonymous review:

- `f4a_local_shuffle.py`: SHA256 `e24875ea269e9a55974c71bf948cb56c5294b561146f9ef87c53742ef1fd0bdf`
- `F4X0412.py`: SHA256 `41e08065e9b64a47d8581f282501f64588e27f0dc96057eca0287dee46bba0a0`
- `01_orf_scan.py`: SHA256 `5040dee67588cb39041d967f64bced4f0578e992ff4f21ef8426781ce500bd04`
- `02_rpkm_analysis.py`: SHA256 `df8983d0d88b53aa9e9eae005b0f610915b4fad4a38172df2a354df7fb5315cb`
- `run_moe_InvaRNA0412.py`: SHA256 `2e74417210e72c4815c502f25d57f762fda1cca587c23771be7d162e20d36590`

NEEDS_AUTHOR_CONFIRMATION: a saved `F4A_InvaRNA0412_stats.csv` predates the final
local-shuffle outputs and contains the earlier whole-region-shuffle values for RBP
and miRNA. The later figure script explicitly loads `_f4a_local_s0/s1.csv`; the
repository therefore preserves and recomputes the local-shuffle result, without
silently treating the stale statistics file as final.

NEEDS_AUTHOR_CONFIRMATION: the recovered case-study summary contains two complete
four-expert rows for NDUFA4 (52 rows total for 12 symbols), while the plotting loader
selects the first summary row but concatenates all matching background rows. The
files are preserved unchanged; this duplicate should be resolved against the actual
submitted Fig. 4B before release.
