# Known limitations

The repository does not fabricate missing historical implementations or raw data.
Current limitations relevant to independent review are:

- the original 274-species Ensembl release-111 FASTA is not redistributed; users must download the relevant Ensembl mature-mRNA releases and rebuild their own corpus;
- an authenticated Flow-Matching model-training implementation was not recovered, although the frozen pools and the exact UTR-replacement consumption logic are preserved;
- several Fig. 4 experimental panels are reproducible from curated summary tables rather than complete raw measurements, and some tested Fig. 4 construct sequences still require author supply;
- the exact historical all-task Fig. 5 production launcher/config was not recovered; the public release provides a TE/half-life objective, environment/PPO code, and an NGF smoke chain. This workflow does not reproduce the historical candidate-generation trace;
- the recovered 13,653-row NGF trajectory contains 177 sequences whose translated
  protein differs from the first trajectory sequence, whereas the recovered current
  `SequenceEnv` smoke path enforces synonymous actions; the authenticated historical
  source/config needed to resolve this discrepancy was not found;
- Extended Data Figs. 1–5 provide checked panel-level source tables only, without
  figure-drawing code or raw benchmark/training inputs. For Extended Data Fig. 5,
  the published table explicitly resolves the two `MUT3` rows in the source
  workbook; the unreleased historical plotting script still expects an older label;
- CodonBERT and RiboNN are academic/non-commercial components and block an unrestricted public binary/source bundle unless excluded or separately permitted.

The reviewer-facing limitations are summarized here and in
`docs/MODEL_PROVENANCE.md`; internal forensic audit notes are kept outside the
public repository.
