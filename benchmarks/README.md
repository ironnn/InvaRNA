# Benchmarks

This directory evaluates frozen predictions against labels. It does not implement
model inference; prediction-only commands are under `inference/`, while external-model
implementations are under `external_models/`.

| Benchmark | Entry point | Frozen inputs |
|---|---|---|
| Held-out human TE | `heldout_human_te/verify_external_models.py` | `assets/benchmark_data/human_te/` |
| RPFdb multi-species | `evaluate.py` | `assets/benchmark_data/rpfdb/` |
| MPRA | `mpra/predict.py` and `figures/fig3/fig3g_mpra.py` | `assets/benchmark_data/mpra/` |
| Representation transfer | `representation_transfer/fit_probe.py` | `assets/benchmark_data/representation_transfer/` |
| RBH sensitivity | `rbh_sensitivity/audit.py` | `assets/benchmark_data/rbh_sensitivity/` |

RPFdb and MPRA remain external evaluation only and never enter teacher fitting,
synthetic-label construction, checkpoint selection, or student training.
