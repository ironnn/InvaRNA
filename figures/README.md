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

Extended Data Figs. 1–5 are a data-only release: see
`figures/extended_data/README.md` for the panel source-table map.
