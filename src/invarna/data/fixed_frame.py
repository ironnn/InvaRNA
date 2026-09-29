"""Sequence framing used by the selected Stage-2 training and inference code.

Short 5' UTRs are left-padded so the CDS begins at position 1,000. The recovered
training code does not left-truncate 5' UTRs longer than 1,000 nt; preserving that
behavior is necessary for checkpoint-compatible inference.
"""

FIXED_CDS_START = 1000
TOTAL_LENGTH = 10000


def align_and_pad(sequence: str, utr5_size: int) -> str:
    """Apply the recovered Stage-2 padding/truncation rule and return 10,000 nt."""
    sequence = str(sequence).upper().replace("U", "T")
    left_padding = max(FIXED_CDS_START - int(utr5_size), 0)
    return ("N" * left_padding + sequence)[:TOTAL_LENGTH].ljust(TOTAL_LENGTH, "N")
