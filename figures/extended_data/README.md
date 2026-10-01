# Extended Data Figures 1–5: panel source data

These directories provide **tables only** for the five author-designated Extended
Data figures. No plotting code or claim of pixel-identical figure regeneration is
provided.

| Figure | Panel | Data table(s) | Meaning |
|---|---|---|---|
| Ext. 1 | A | `ext_fig1/data/panel_a_corpus_summary.csv`, `panel_a_top_species.csv` | Corpus totals and top-species transcript counts |
| Ext. 1 | B | `ext_fig1/data/panel_b_region_availability.csv`, `panel_b_transcript_length_ecdf.csv`, `panel_b_transcript_length_ecdf_meta.csv` | Region counts, transcript-length ECDF and cutoff annotations |
| Ext. 1 | C | `ext_fig1/data/panel_c_fixed_frame_examples.csv`, `panel_c_fixed_frame_qc.csv` | Illustrative fixed-frame layouts and corpus-level truncation rates |
| Ext. 1 | D | `ext_fig1/data/panel_d_moe_loss_curves.csv` | With/without Motif-MoE training-loss curves |
| Ext. 2 | A | `ext_fig2/data/panel_a_crispr_on_target.csv` | Test Spearman correlation |
| Ext. 2 | B | `ext_fig2/data/panel_b_crispr_off_target.csv` | Test Spearman correlation |
| Ext. 2 | C | `ext_fig2/data/panel_c_rna_switches.csv` | Mean test Spearman across the three RNA-switch targets |
| Ext. 2 | D | `ext_fig2/data/panel_d_rna_modification.csv` | Test macro AUC |
| Ext. 3 | A | `ext_fig3/data/panel_a_normalized_mutation_rates.csv`, `panel_a_region_wise_mutation_counts.csv` | K80 and matched-random mutation rates and regional counts |
| Ext. 3 | B | `ext_fig3/data/panel_b_hamming_distances.csv` | Hamming-distance ECDF inputs |
| Ext. 3 | C | `ext_fig3/data/panel_c_tdc_label_shifts.csv` | TDC shift-density inputs relative to measured WT TE |
| Ext. 4 | Heatmap and mean | `ext_fig4/data/cross_species_ablation_spearman.csv` | One row per ablation model: five full-precision species Spearman values and their five-species mean |
| Ext. 5 | Full NGF screen | `ext_fig5/data/ngf_elisa_all_variants.csv` | REF and MUT1–MUT20; unmodified and N1-methylpseudouridine-modified ELISA replicates, means and sample SDs |

Provenance: Ext. 1 tables are copied without numerical modification from
`final_figures/sup/s1/data/panel_*.csv`; Ext. 3 tables likewise from
`final_figures/sup/s3/data/panel_*.csv`. Ext. 2 tables are extracted from
`final_figures/sup/s2/results/beacon_results.csv`, using the **test** split,
excluding 3UTRBERT (not plotted), and retaining the published bar order. For
Ext. 2C, each value is the arithmetic mean of the three test Spearman values
for `t0`, `t1`, and `t2`. The original source files remain outside this repository.

Ext. 4 is taken from the **current manuscript figure**
`final_paper0806/final_paper/figures/ext4.pdf`, which is byte-identical to
`InvaRNA/benchmark/cross_species/ablation_rerun_20260704/fig3E_cross_species_ablation_spearman_revised_no_fly.pdf`.
Its numeric source is that directory's
`fig3E_six_model_cross_species_metrics.csv`. The single Ext. 4 table retains exactly the
five species displayed in the manuscript: C. elegans, D. rerio, H. sapiens,
M. musculus, and R. norvegicus. Its mean column is recalculated from those five
values, not copied from the historical six-species `mean` row; D. melanogaster
is intentionally excluded. Values are unrounded, while the figure displays
two decimals in the heatmap and three decimals beside the mean bars.

Ext. 5 uses the current manuscript's `ext51.pdf`, byte-identical to
`final_figures/F5/F5C/NGF_ELISA_AllVariants_grouped.pdf`. Its source is the
`板 1 - Sheet1` worksheet in `final_figures/F5/F5C/20251219 hNGF.xlsx`,
columns 1, 2, 3, 5, and 6 of zero-indexed rows 136–157. The current workbook
has **two** rows labelled `MUT3`: Excel row 138 is the displayed industrial
reference renamed to `MUT3`, and row 141 is the separate ordinary `MUT3` row
excluded from the figure. The table records each selected source Excel row so
the figure's choice is unambiguous. `group` reproduces the figure's reference,
advanced-to-stability, and other-design colors; the two condition means are
the bar heights, the sample SDs (`ddof=1`) are error bars, and the four
replicate columns are the plotted points. The REF condition means are the two
horizontal reference lines. The historical plotting script still uses the old
`MA-hek-3` label and will not reproduce the figure from the relabelled Excel
without this duplicate-name correction; this CSV is the data-only release.

The source figure files, respectively, are
`final_figures/sup/s1/extended_data_fig1_pretraining_qc.pdf`,
`final_figures/sup/s2/results/beacon_compare_v2.pdf`, and
`final_figures/sup/s3/extended_data_fig4_synthetic_variant_qc.pdf`. The last
filename reflects an older figure number; it is **Extended Data Fig. 3** in the
current manuscript. Source figures, backup files, raw training logs, benchmark
embeddings, large sampled caches, and plotting scripts are not distributed here.
