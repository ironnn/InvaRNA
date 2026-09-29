#!/usr/bin/env bash
set -euo pipefail

echo "The historical 274-species pretraining corpus is not distributed by this repository." >&2
echo "Download the required sequence and annotation files directly from Ensembl release 111." >&2
echo "The exact species manifest and historical mature-mRNA reconstruction script are not provided." >&2
echo "See assets/README.md and docs/DATA_PROVENANCE.md for the reproducibility boundary." >&2
exit 1
