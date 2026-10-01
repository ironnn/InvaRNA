# Third-party software and model notices

## License scope

The repository-level MIT license applies only to material authored for the
InvaRNA project unless a file or directory says otherwise. It does **not**
relicense third-party source code, model configurations, pretrained weights, or
fine-tuned derivatives. Each third-party component remains subject to its
upstream license and notices.

The authoritative machine-readable inventory for copied baseline components is
[`external_models/THIRD_PARTY_LICENSES.tsv`](external_models/THIRD_PARTY_LICENSES.tsv).
InvaRNA-authored integration code under `external_models/common/`,
`external_models/training/`, and `external_models/inference/` is covered by the repository
MIT license unless an individual file says otherwise.

## Components distributed with the repository or asset archive

| Component | Copied artifact and governing terms | Preserved evidence |
|---|---|---|
| CodonBERT | Sanofi academic-research/non-commercial artifact and software notices; not MIT and not an unrestricted open-source grant. | `external_models/CodonBERT/ARTIFACT_LICENSE.md` and `SOFTWARE_LICENSE.md` |
| DNABERT-2 | Apache-2.0. | `external_models/DNABERT2/LICENSE` |
| Orthrus | MIT, with the upstream copyright notice retained. | `external_models/Orthrus/LICENSE.md` |
| RNA-FM | The copied artifact is the MultiMolecule adaptation, whose model card declares AGPL-3.0 and whose License FAQ applies to its distributed weights. The original RNA-FM repository's MIT license does not replace the copied adaptation's declared terms. | `external_models/RNA_FM/license.md`, `license-faq.md`, and the model card |
| UTR-LM | GPL-3.0. | `external_models/UTR_LM/LICENSE` |
| Evo 2 | Apache-2.0; upstream NOTICE and AUTHORS are also retained. The archived Evo2 checkpoint is identified by its exact Hugging Face source. | `external_models/Evo2/LICENSE`, `NOTICE`, and `AUTHORS` |
| LucaOne | The copied source package follows the official repository's Apache-2.0 license. The exact `LucaOne-gene-step36.8M` Hugging Face artifact separately declares MIT in its model card and publishes no standalone license file; this distinction is preserved rather than silently choosing one license for both artifacts. | `external_models/LucaOne/source/LICENSE` and checkpoint `LICENSE_SOURCE.md` |
| mRNABERT | Apache-2.0, as declared by the exact model card and official repository. | `external_models/mRNABERT/LICENSE` and the model card |
| RiboNN | Separate Sanofi academic-research/non-commercial notices govern the software and model weights. Neither is relicensed as MIT. | `external_models/RiboNN/RiboNN-1.0.0/LICENSE` and `MODEL WEIGHTS LICENSE.txt` |

The RNA-FM License FAQ states obligations for conveyed upstream and modified model
weights, including corresponding source. Consequently, its
fine-tuned TE checkpoints must travel with the AGPL text, the FAQ, and the
corresponding benchmark source. GPL/AGPL components remain separable
third-party programs; this repository makes no claim that its MIT license
overrides their copyleft terms.

## Redistribution gate

CodonBERT and RiboNN grant use/copy rights only for academic research and
non-commercial use by academic or non-profit recipients. Their copied source
and weights must therefore **not** be placed in an unrestricted public-release
bundle unless the authors obtain permission confirming that distribution or
exclude those artifacts and provide upstream download instructions. An
anonymous-review deposit must preserve the notices and restrict access/use to
the scope granted by those notices.

Run the inventory check before creating any deposit:

```bash
python external_models/audit_licenses.py
```

For an unrestricted public release, use the strict gate. It intentionally
fails while restricted CodonBERT/RiboNN artifacts remain in the distribution:

```bash
python external_models/audit_licenses.py --strict-public
```

The seven native checkpoints tracked as compressed files in Git contain no
third-party baseline weights or historical training data. Any future third-party
weight distribution requires a separate license review and the applicable
notices alongside those weights.

## Provenance checked on 2026-09-23

- CodonBERT: https://github.com/Sanofi-Public/CodonBERT
- DNABERT-2: https://github.com/MAGICS-LAB/DNABERT_2
- Orthrus: https://github.com/bowang-lab/Orthrus
- RNA-FM MultiMolecule artifact: https://huggingface.co/multimolecule/rnafm
- UTR-LM: https://github.com/a96123155/UTR-LM
- Evo 2 source and checkpoint: https://github.com/ArcInstitute/evo2 and https://huggingface.co/arcinstitute/evo2_7b
- LucaOne source and exact checkpoint: https://github.com/LucaOne/LucaOne and https://huggingface.co/LucaGroup/LucaOne-gene-step36.8M
- mRNABERT exact checkpoint: https://huggingface.co/YYLY66/mRNABERT
- RiboNN: https://github.com/Sanofi-Public/RiboNN

This inventory records upstream terms and release safeguards; it is not legal
advice.
