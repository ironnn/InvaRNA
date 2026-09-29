# Figure 5

The public plotting bundle retains only the checked tables required for the displayed
F5C, F5D, and 20x F5E panels. Candidate histories, raw experimental workbooks, and
design-training tables are excluded. The small WT starting sequences and the RL smoke
fixture remain in `assets/sequences/` and `assets/smoke/`.

Redraw the Fig. 5C panel from the frozen public tables with:

```bash
python figures/fig5/plot_f5c.py
```

The checked-table redraw is written to `figures/figure_subplot_check/fig5/NGF_combined.pdf`.

The current-paper 20x Fig. 5E panel is rendered with:

```bash
python figures/fig5/plot_f5e_checked.py
```

and the checked Fig. 5D panel with:

```bash
python figures/fig5/plot_f5d_checked.py
```

The industrial reference formerly called `MA-hek-3` is displayed and released as
`MUT3`. The separate ordinary `mut3` Excel row is not used by this panel and is not
included in the standalone plotting tables.
