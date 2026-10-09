# Figure 5

The public plotting bundle retains only the checked tables required for the displayed
F5C, F5D, and 20x F5E panels. Candidate histories, raw experimental workbooks, and
design-training tables are excluded. The complete BLNK, HBB, and NGF construct
sequences are provided through the independently uploaded Supplementary Data 2,
not this Git repository. The WT starting sequences and RL smoke fixture remain
in `assets/sequences/` and `assets/smoke/`.

The supported one-command redraw is `bash figures/fig5/reproduce.sh`, which runs
the three C/D/E plotting scripts below. No full-panel raw-data audit entry point
is distributed because the F5A/B historical inputs are not in this Git checkout.

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

## RL workflow

The public sequence-scoring objective in `src/invarna/design/reward.py` is:

```text
R = 0.70 * robust_TE + 0.30 * z_HL
robust_TE = mean(z_TE) - penalty
penalty = 0.15 * std(z_TE) if std(z_TE) > 0.5, otherwise 0
```

The TE ensemble uses `w0`, `w1`, `w2`, and `final_tdc` with equal weights;
the half-life predictor is `hl_new`. The BLNK/HBB configurations specify
WT-centered predictor z-scores clipped to [-3, 3]. Callers must supply the frozen
normalization and WT references before invoking the reward helper.

The environment uses synonymous CDS actions, start/stop protection, and local
safety checks for restriction sites, miRNA seeds, ARE motifs, and G4 patterns
within ±15 nt of the mutation site. Rejected actions consume a step and incur -0.05. An
optional minimum-edit constraint gives a terminal reward of -10 when unmet; its
default is zero required edits. These are environment constraints, separate from
the sequence-scoring objective above.

Run the CPU integration smoke from the repository root:

```bash
python tests/smoke/test_fig5_rl_pipeline.py --output-dir results/fig5_rl_smoke
```

The smoke uses deterministic position-specific toy scores for four TE outputs
and one half-life output, passed through the public reward helper. It exercises
the environment, rollout, GAE, PPO update, checkpoint save/load, and synonymous
candidate export. The toy scores have no biological interpretation and do not
load the predictor checkpoints. The public configurations and smoke demonstrate
the design workflow; they do not reproduce historical manuscript
candidate-generation traces.
