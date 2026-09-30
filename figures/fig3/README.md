# Figure 3

`reproduce.sh` redraws the current-paper combined figure from the checked tables in
`assets/manuscript_figures/full/fig3_current_nm/`. It does not load an external model
checkpoint. The Fig. 3E ablation values are stored in
`F3E_ablation_plot_data.csv` alongside the other panel sources. Run
`python tests/test_manuscript_values.py` to check current Fig. 3E/F/G values against
the manuscript-rounded targets. The individual scripts remain available for
benchmark re-analysis, but are not the publication plotting entry point.
