"""Local safety and CDS translation constraints used during sequence design."""

import re

from Bio.Seq import Seq

FORBIDDEN_ENZYMES = ("GAATTC", "GGATCC", "AAGCTT", "GCGGCCGC", "CTCGAG", "GCTAGC")
MIRNA_SEEDS = (
    "CACTCCA", "TGCCAAA", "ACTGACA", "GTTCTCA", "AGCATTA", "ACACTAC",
    "TGCCTTA", "TCGATAC", "ACTGTGA", "CTACCTC", "TGCTGCT", "TAAGCTA", "CGCACAG",
)
ARE_MOTIFS = ("ATTTATTTA", "TATTTAT")
G4_PATTERN = re.compile(r"(G{3,}\w{1,7}){3,}G{3,}")
LOCAL_WINDOW = 15
REJECTION_PENALTY = -0.05


def passes_motif_constraints(sequence: str) -> bool:
    sequence = str(sequence).upper().replace("U", "T")
    motifs = FORBIDDEN_ENZYMES + MIRNA_SEEDS + ARE_MOTIFS
    return not any(motif in sequence for motif in motifs) and G4_PATTERN.search(sequence) is None


def preserves_translation(wild_type_cds: str, candidate_cds: str) -> bool:
    return str(Seq(wild_type_cds).translate()) == str(Seq(candidate_cds).translate())
