"""
Orthrus 6-track encoder.

Orthrus is a non-HF Mamba model whose input is NOT token ids but a
(L, 6) channels-last float tensor:
  channels 0-3 : one-hot ACGU/ACGT  (order A,C,G,T — matches orthrus/gk_utils.py:seq_to_oh)
  channel  4   : CDS track   (1 inside CDS, else 0)
  channel  5   : splice track (1 at splice sites, else 0)

For bare RNA sequences with no CDS/splice annotation we set channels 4-5 to 0.
This mirrors orthrus/finetune_dataloader.py (n_tracks=6: one-hot + cds + splice).
"""
import numpy as np
import torch


def seq_to_6track(seq: str) -> np.ndarray:
    """RNA/DNA string -> (L, 6) float32, channels-last. Channels 4-5 (cds/splice) = 0."""
    seq = str(seq).upper().replace("U", "T")
    L = len(seq)
    arr = np.zeros((L, 6), dtype=np.float32)
    idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, b in enumerate(seq):
        j = idx.get(b)
        if j is not None:
            arr[i, j] = 1.0
    # channels 4 (cds) and 5 (splice) left as 0 for bare sequences
    return arr


def encode_batch(seqs, device="cpu"):
    """List[str] -> (padded tensor (B, Lmax, 6), lengths tensor (B,))."""
    tracks = [seq_to_6track(s) for s in seqs]
    lengths = torch.tensor([t.shape[0] for t in tracks], dtype=torch.long)
    Lmax = int(lengths.max().item()) if len(tracks) else 0
    batch = np.zeros((len(tracks), Lmax, 6), dtype=np.float32)
    for i, t in enumerate(tracks):
        batch[i, : t.shape[0]] = t
    x = torch.from_numpy(batch).to(device)          # (B, Lmax, 6) channels-last
    return x, lengths.to(device)
