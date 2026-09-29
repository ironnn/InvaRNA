# Known limitations

The repository does not fabricate missing historical implementations or raw data.
Current limitations relevant to independent review are:

- the original 274-species Ensembl release-111 FASTA is not redistributed; users must download the relevant Ensembl mature-mRNA releases and rebuild their own corpus;
- an authenticated Flow-Matching model-training implementation was not recovered, although the frozen pools and the exact UTR-replacement consumption logic are preserved;
- several Fig. 4 experimental panels are reproducible from curated summary tables rather than complete raw measurements, and some tested Fig. 4 construct sequences still require author supply;
- the exact historical all-task Fig. 5 production launcher/config was not recovered; the copied early TE/half-life PPO source and NGF smoke chain are retained without rewriting the algorithm;
- the recovered 13,653-row NGF trajectory contains 177 sequences whose translated
  protein differs from the first trajectory sequence, whereas the recovered current
  `SequenceEnv` smoke path enforces synonymous actions; the authenticated historical
  source/config needed to resolve this discrepancy was not found;
- Extended Data Figs. 1–5 currently have explicit placeholder entry points only;
  their final panel-specific source tables were not approved for the review archive,
  so the repository does not claim those panels are independently reproduced;
- CodonBERT and RiboNN are academic/non-commercial components and block an unrestricted public binary/source bundle unless excluded or separately permitted.

The reviewer-facing limitations are summarized here and in
`docs/MODEL_PROVENANCE.md`; internal forensic audit notes are kept outside the
public repository.
