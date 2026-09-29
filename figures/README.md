# Manuscript figure reproduction

This is the primary reviewer entry point. Figure scripts orchestrate frozen assets,
inference, benchmark calculations, statistical analysis, and plotting. They do not
contain a second copy of the model or training algorithms.

```bash
bash figures/fig2/reproduce.sh
bash figures/fig3/reproduce.sh
bash figures/fig4/reproduce.sh
bash figures/fig5/reproduce.sh
```

Panel-level provenance and unresolved raw-data limitations are listed in
`docs/FIGURE_AUDIT.csv` and `docs/KNOWN_LIMITATIONS.md`.
