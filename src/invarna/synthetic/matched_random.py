"""Region-wise matched-random control used in the TE ablation."""

from __future__ import annotations

import random
from Bio.Data import CodonTable
from Bio.Seq import Seq

DNA = "ACGT"


def _replace(values, count, rng, start=0, stop=None):
    stop = len(values) if stop is None else stop
    eligible = [i for i in range(start, stop) if values[i] in DNA]
    if count > len(eligible):
        raise ValueError("requested burden exceeds eligible positions")
    for position in rng.sample(eligible, count):
        values[position] = rng.choice([base for base in DNA if base != values[position]])


def _synonymous_table():
    table = CodonTable.unambiguous_dna_by_id[1]
    codons = [a + b + c for a in DNA for b in DNA for c in DNA]
    amino_acid = {codon: str(Seq(codon).translate(table=table)) for codon in codons}
    return {codon: [other for other in codons if other != codon and amino_acid[other] == aa] for codon, aa in amino_acid.items()}


SYNONYMS = _synonymous_table()


def mutate_matched(mrna, utr5_size, cds_size, utr3_size, utr5_diff, cds_diff, utr3_diff, *, cds_mode="free", seed=42):
    """Randomize positions/bases while matching K80 regional burdens.

    ``free`` protects CDS start/stop codons. ``strict`` preserves translation,
    but can overshoot nucleotide burden when a synonymous codon differs at two
    bases; this is the recorded behavior of the original strict-control branch.
    """
    if len(mrna) != utr5_size + cds_size + utr3_size:
        raise ValueError("region lengths do not sum to mRNA length")
    if cds_mode not in {"free", "strict"}:
        raise ValueError("cds_mode must be free or strict")
    rng = random.Random(seed)
    utr5 = list(mrna[:utr5_size])
    cds = list(mrna[utr5_size:utr5_size + cds_size])
    utr3 = list(mrna[utr5_size + cds_size:])
    _replace(utr5, utr5_diff, rng)
    _replace(utr3, utr3_diff, rng)
    if cds_mode == "free":
        _replace(cds, cds_diff, rng, 3, max(3, len(cds) - 3))
    else:
        starts = [i for i in range(3, len(cds) - 3, 3) if SYNONYMS.get("".join(cds[i:i + 3]))]
        rng.shuffle(starts)
        changed = 0
        for start in starts:
            if changed >= cds_diff:
                break
            old = "".join(cds[start:start + 3])
            new = rng.choice(SYNONYMS[old])
            cds[start:start + 3] = new
            changed += sum(a != b for a, b in zip(old, new))
        if changed < cds_diff:
            raise ValueError("not enough synonymous alternatives")
    return "".join(utr5 + cds + utr3)
