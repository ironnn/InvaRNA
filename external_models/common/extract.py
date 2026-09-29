"""
Embedding extraction via the unified backbone registry.

Single code path for all 9 backbones (the same registry used for full fine-tuning):
loads the base encoder, runs forward_hidden -> (B, L, H), then pools.

Two modes:
  - whole-sequence: extract_embeddings() -> per-token (L,H) or pooled (H,)
  - region-aware:   embed_regions() -> per-region mean+max concat, driven by spec.regions
                    (codonbert->cds, rest->utr5/cds/utr3). Mirrors the
                    InvaRNA ablation (pipeline_k80/4_extract_embedding.py) and the region
                    convention in sft/train_te_*.py.

Returns a dict per call so run_embedding.py / programmatic callers can use it directly.
"""
import torch

from external_models.common.backbones import get_spec


@torch.no_grad()
def extract_embeddings(backbone_key, seqs, device="cuda:0", pooling="none", max_len=None):
    """
    Args:
        backbone_key: one of unified.backbones.ALL_KEYS
        seqs: str | list[str]
        device: 'cuda:0' / 'cpu'
        pooling: 'none' (per-token (L,H)) | 'mean' | 'max' | 'cls'
        max_len: truncation length; None -> spec.default_max_len (1024, lucaone 1280)
    Returns:
        list of tensors (one per sequence): (L, H) if pooling='none' else (H,)
    """
    if isinstance(seqs, str):
        seqs = [seqs]
    spec = get_spec(backbone_key)
    if max_len is None:
        max_len = spec.default_max_len
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    backbone = spec.load_backbone(device=str(dev), dtype=None)
    if hasattr(backbone, "eval"):
        backbone.eval()
    tokenizer = spec.load_tokenizer()

    results = []
    for s in seqs:
        inputs = spec.encode(tokenizer, [s], max_len)
        inputs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        hidden = spec.forward_hidden(backbone, inputs)  # (1, L, H)
        h = hidden.squeeze(0).float().cpu()             # (L, H)
        mask = inputs.get("attention_mask", None)
        if pooling == "mean":
            if mask is not None:
                m = mask.squeeze(0).float().cpu().unsqueeze(-1)
                h = (h * m).sum(0) / m.sum(0).clamp(min=1.0)
            else:
                h = h.mean(0)
        elif pooling == "max":
            h = h.max(0)[0]
        elif pooling == "cls":
            h = h[0]
        results.append(h)
    return results


# ───────────────────────── region-aware extraction ──────────────────────────
def slice_region(name, mrna, u5, cds, u3):
    """Slice one region substring from a full mRNA. Sizes are in nucleotides and
    mrna == utr5 + cds + utr3 (see sft/train_te_*.py extract_region)."""
    if name == "utr5":
        return mrna[:u5]
    if name == "cds":
        return mrna[u5:u5 + cds]
    if name == "utr3":
        return mrna[u5 + cds:]
    raise ValueError(f"unknown region '{name}' (expected utr5|cds|utr3)")


def load_backbone_and_tok(backbone_key, device="cuda:0"):
    """Load backbone + tokenizer once, to reuse across many sequences (batch extraction)."""
    spec = get_spec(backbone_key)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    backbone = spec.load_backbone(device=str(dev), dtype=None)
    if hasattr(backbone, "eval"):
        backbone.eval()
    tokenizer = spec.load_tokenizer()
    return spec, backbone, tokenizer


@torch.no_grad()
def embed_whole(spec, backbone, tokenizer, seq, device="cuda:0", max_len=None):
    """Whole-sequence embedding: encode+forward one sequence, pool with mean+max
    concatenated. Returns a (2*H,) float32 cpu tensor. Empty seq -> zeros(2*H).

    This is the shared pooling primitive: embed_regions() calls it per region
    substring, and the CSV/whole-seq extractor calls it once per sequence.

    max_len: None -> spec.default_max_len (1024, lucaone 1280).
    """
    if max_len is None:
        max_len = spec.default_max_len
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    H = spec.hidden_dim
    seq = str(seq).upper()
    if len(seq) == 0:
        return torch.zeros(2 * H)
    inputs = spec.encode(tokenizer, [seq], max_len)
    inputs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    h = spec.forward_hidden(backbone, inputs).squeeze(0).float().cpu()   # (L, H)
    mask = inputs.get("attention_mask", None)
    if mask is not None:
        m = mask.squeeze(0).float().cpu().unsqueeze(-1)                   # (L, 1)
        mean = (h * m).sum(0) / m.sum(0).clamp(min=1.0)
        mx = h.masked_fill(m == 0, float("-inf")).max(0)[0]
        mx = torch.nan_to_num(mx, neginf=0.0)                            # all-masked -> 0
    else:
        mean, mx = h.mean(0), h.max(0)[0]
    return torch.cat([mean, mx])                                         # (2H,)


@torch.no_grad()
def embed_regions(spec, backbone, tokenizer, mrna, u5, cds, u3, device="cuda:0", max_len=None):
    """Region-aware embedding: for each region in spec.regions, slice the substring,
    encode+forward it through the backbone, and pool with mean+max concatenated
    (matches InvaRNA ablation pipeline_k80/4_extract_embedding.py).

    Each region is run as its OWN forward pass on its substring — the only
    tokenizer-agnostic method (BPE/codon/k-mer tokenizers have no 1:1 nt<->token map).

    max_len: None -> spec.default_max_len (1024, lucaone 1280).
    Empty region -> zeros(2*H). Returns a (len(spec.regions) * 2 * H,) float32 cpu tensor.
    """
    mrna = str(mrna).upper()
    u5, cds, u3 = int(u5), int(cds), int(u3)
    out = [embed_whole(spec, backbone, tokenizer, slice_region(name, mrna, u5, cds, u3),
                       device=device, max_len=max_len)
           for name in spec.regions]
    return torch.cat(out)                                                    # (len(regions)*2H,)
