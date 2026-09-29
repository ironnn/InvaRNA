# Data generation

This directory records how synthetic sequences and their training labels were produced.
It never serves as the location of manuscript-frozen datasets; those live under
`assets/training_data/`.

- `evolutionary_variants/`: K80-derived local variants, which may contain multiple substitutions, and matched-random controls.
- `flow_matching/`: non-local UTR replacement generation. No WT anchoring or distance scaling is applied.
- `label_generation/`: privileged-teacher feature extraction/scoring and WT-anchored TDC label construction.

The recovered repository contains the Flow-Matching pool-consumption/generation step,
but not an authenticated Flow-Matching model-training implementation. No replacement
implementation is fabricated; see `docs/KNOWN_LIMITATIONS.md`.
