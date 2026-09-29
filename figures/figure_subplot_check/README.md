# Canonical figure checks

This directory contains one submission-facing redraw for each figure check. The
files are tied to the current paper data, rather than to superseded exploratory
plots.

- `fig2/` contains the final combined PDF copied from the designated submission
  figure package.
- `fig3/` contains the checked-data combined redraw plus the current ablation,
  no-fly cross-species panel, and MPRA panel G checks. Its companion source
  tables are `assets/manuscript_figures/full/fig3_current_nm/`; the combined
  redraw reuses the final-submit plotting layout via
  `python figures/fig3/reproduce_combined_checked.py`.
- `fig4/` contains the F4X0423 submission-basis panels.
- `fig5/` contains the numeric audit and the current-paper F5B–F5E panels.
  F5C is regenerated from the unmodified-mRNA tables, F5D from the checked
  western-blot table with the industrial reference labelled `MUT3`, and F5E
  contains only the paper's 20x RGC panel.

The obsolete May-2022 Fig. 3 panels and duplicate simplified Fig. 4 outputs were
removed so that there is no competing version in the repository. A full Fig. 2
GPU replay is not represented here because it requires a free inference GPU; no
stale replay output is retained.
