"""
Backbone registry — the single source of truth for how each embedding backbone is
loaded, tokenized, and run to produce a (B, L, H) hidden-state tensor.

Both modes use this registry:
  - embedding extraction (run_embedding.py)
  - full-parameter fine-tuning (finetune_unified.py, via unified/model.py)

Every backbone exposes the same 5-capability contract via BackboneSpec:
  load_backbone(model_dir, device, dtype) -> nn.Module   # the base ENCODER (no head)
  load_tokenizer(model_dir)               -> tokenizer | None
  encode(tokenizer, seqs, max_len)        -> dict of input tensors (on cpu)
  forward_hidden(backbone, inputs)        -> (B, L, H) hidden states
  hidden_dim                              -> int

This file is ENVIRONMENT-AGNOSTIC (pure torch + lazy imports). Each backbone is only
importable in its own conda env; we import lazily inside the load fns so that importing
this module never fails for backbones you're not using.

Per-backbone conda env (see README):
  rnafm            -> threeutrlm_env       (multimolecule + transformers 4.50)
  codonbert, dnabert2 -> agent2            (transformers 4.57)
  mrnabert         -> mrnabert_env
  lucaone          -> lucaone
  orthrus          -> caduceus_env         (mamba_ssm)
  evo2             -> mamballm/lucaone      (vortex)  [embedding only for now]
  invarna          -> mamballm              (InvaRNA Caduceus; embedding only)
"""
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch

BASELINES_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = BASELINES_ROOT.parent
EXTERNAL_ASSET_ROOT = Path(
    os.environ.get(
        "INVARNA_EXTERNAL_ASSET_ROOT",
        REPO_ROOT / "assets" / "checkpoints" / "external" / "pretrained",
    )
)
MODEL_ROOT = EXTERNAL_ASSET_ROOT / "models"

# InvaRNA repo is a SIBLING of llm_models/ (not under models/); env-overridable.
INVARNA_ROOT = os.environ.get("INVARNA_ROOT", str(REPO_ROOT))


@dataclass
class BackboneSpec:
    key: str
    model_dir: str
    hidden_dim: int
    env: str
    seq_kind: str  # 'rna' | 'dna' | 'cds' | 'gene'
    load_backbone: Callable
    load_tokenizer: Callable
    encode: Callable
    forward_hidden: Callable
    supports_full_ft: bool = True
    # head_mode: 'native'  -> model ships its own regression/cls head
    #                         (AutoModelForSequenceClassification or multimolecule SequencePrediction)
    #            'shared'  -> no native head; use unified SharedSeqClsHead
    head_mode: str = "native"
    # how the native head is built (only used when head_mode == 'native'):
    #   'hf_auto'        -> transformers.AutoModelForSequenceClassification.from_pretrained
    #   'multimolecule'  -> multimolecule.AutoModelForSequencePrediction.from_pretrained
    native_loader: str = "hf_auto"
    notes: str = ""
    # which mRNA region(s) this backbone embeds (region-aware extraction).
    # region-specific models override (codonbert->cds); the rest
    # embed all three. Source of truth mirrors sft/train_te_*.py extract_region().
    regions: tuple = ("utr5", "cds", "utr3")
    # default embedding truncation length (tokens). Used by extract.py /
    # run_embedding* when the caller doesn't pass --max_len explicitly. Most models
    # use 1024; lucaone uses 1280 to match its training (sft/train_te_lucaone.py).
    default_max_len: int = 1024


# ───────────────────────────── helpers ──────────────────────────────────────
def _hf_last_hidden(backbone, inputs):
    """Standard HF: outputs.last_hidden_state, with tuple fallback (DNABERT2/mRNABERT)."""
    out = backbone(**inputs)
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state
    if isinstance(out, (tuple, list)):
        return out[0]
    # BaseModelOutput-like
    return out[0]


def _hf_encode(tokenizer, seqs, max_len):
    enc = tokenizer(
        list(seqs), return_tensors="pt", padding="longest",
        truncation=True, max_length=max_len, add_special_tokens=True,
    )
    return dict(enc)


# ───────────────────────────── RNA-FM (multimolecule) ─────────────
def _mm_load(model_dir):
    def _f(device="cuda:0", dtype=None):
        from multimolecule import AutoModel as MMAutoModel
        m = MMAutoModel.from_pretrained(model_dir, local_files_only=True)
        if dtype is not None:
            m = m.to(dtype)
        return m.to(device)
    return _f


def _mm_tok(model_dir):
    def _f():
        from multimolecule import RnaTokenizer
        return RnaTokenizer.from_pretrained(model_dir, local_files_only=True)
    return _f


# ───────────────────────────── CodonBERT ─────────────────────────────────────
def _codonbert_load(model_dir):
    def _f(device="cuda:0", dtype=None):
        from transformers import BertModel
        m = BertModel.from_pretrained(model_dir, dtype=(dtype or torch.float32))
        return m.to(device)
    return _f


def _codonbert_tok(model_dir):
    def _f():
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=False)
    return _f


def _codonbert_encode(tokenizer, seqs, max_len):
    """CodonBERT tokenizes a LIST of codons per sequence.

    CodonBERT's vocab is RNA (AUG, AAU, ...), so sequences must use U, not T.
    Converting T->U here; feeding DNA codons (with T) makes every T-containing
    codon an [UNK] (only the 27/64 T-free codons survive), which silently destroys
    ~half the signal. See vocab.txt: all 64 codons are spelled with U.
    """
    batch_codons = []
    for s in seqs:
        s = str(s).upper().replace("T", "U")
        codons = [s[i:i + 3] for i in range(0, len(s) - len(s) % 3, 3)] or ["AAA"]
        batch_codons.append(codons)
    enc = tokenizer(
        batch_codons, is_split_into_words=True, return_tensors="pt",
        padding="longest", truncation=True, max_length=max_len, add_special_tokens=True,
    )
    return dict(enc)


# ───────────────────────────── DNABERT2 ──────────────────────────────────────
def _trust_remote_load(model_dir):
    def _f(device="cuda:0", dtype=None):
        from transformers import AutoModel
        m = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        if dtype is not None:
            m = m.to(dtype)
        return m.to(device)
    return _f


def _auto_tok(model_dir, trust=True):
    def _f():
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=trust)
    return _f


def _dna_encode(tokenizer, seqs, max_len):
    seqs = [str(s).upper().replace("U", "T") for s in seqs]
    return _hf_encode(tokenizer, seqs, max_len)


def _dnabert2_load(model_dir):
    """DNABERT2: force the PyTorch attention path (nonzero attn dropout) so training
    avoids the flash-attn Triton kernel, which is incompatible with triton>=3.x
    (`dot() got an unexpected keyword argument 'trans_a'`). See bert_layers.py:161."""
    def _f(device="cuda:0", dtype=None):
        from transformers import AutoModel, AutoConfig
        cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        # nonzero -> PyTorch self-attention (bert_layers.py:161), bypasses Triton kernel
        if getattr(cfg, "attention_probs_dropout_prob", 0.0) == 0.0:
            cfg.attention_probs_dropout_prob = 0.1
        m = AutoModel.from_pretrained(model_dir, config=cfg, trust_remote_code=True)
        if dtype is not None:
            m = m.to(dtype)
        return m.to(device)
    return _f


# ───────────────────────────── mRNABERT ──────────────────────────────────────
def _mrnabert_encode(tokenizer, seqs, max_len):
    """mRNABERT expects SPACE-SEPARATED tokens (single nt and/or codons).
    For bare sequences we split into single nucleotides."""
    spaced = []
    for s in seqs:
        s = str(s).upper().replace("U", "T")
        spaced.append(" ".join(list(s)))
    enc = tokenizer.batch_encode_plus(
        spaced, add_special_tokens=True, padding="longest",
        truncation=True, max_length=max_len, return_tensors="pt",
    )
    return dict(enc)


# ───────────────────────────── LucaOne ───────────────────────────────────────
def _ensure_lucaone_importable():
    """`import lucaone` triggers Auto* registration. The package may not be pip-installed;
    fall back to the in-repo source at lucaone_hf_pkg/src."""
    try:
        import lucaone  # noqa: F401
        return
    except ImportError:
        src = str(BASELINES_ROOT / "LucaOne" / "source" / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        import lucaone  # noqa: F401


def _lucaone_load(model_dir):
    def _f(device="cuda:0", dtype=None):
        _ensure_lucaone_importable()
        from transformers import AutoModel
        m = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        if dtype is not None:
            m = m.to(dtype)
        return m.to(device)
    return _f


def _lucaone_tok(model_dir):
    def _f():
        _ensure_lucaone_importable()
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return _f


def _lucaone_encode(tokenizer, seqs, max_len):
    """LucaOne tokenizer needs seq_type='gene' (DNA/RNA)."""
    feats = [tokenizer(str(s), seq_type="gene", add_special_tokens=True,
                       truncation=True, max_length=max_len) for s in seqs]
    keys = feats[0].keys()
    maxlen = max(len(f["input_ids"]) for f in feats)
    out = {}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    for k in keys:
        padval = pad_id if k == "input_ids" else 0
        rows = []
        for f in feats:
            v = list(f[k])
            v = v + [padval] * (maxlen - len(v))
            rows.append(v)
        out[k] = torch.tensor(rows, dtype=torch.long)
    return out


# ───────────────────────────── Orthrus (non-HF Mamba) ────────────────────────
def _orthrus_load(model_dir):
    def _f(device="cuda:0", dtype=None):
        sys.path.insert(0, str(Path(model_dir)))  # so `import orthrus` resolves
        from orthrus.mamba import MixerModel
        m = MixerModel(d_model=512, n_layer=6, input_dim=6)
        sd = torch.load(os.path.join(model_dir, "assets", "pytorch_model.bin"),
                        map_location="cpu")
        if "state_dict" in sd:
            sd = {k.split("model.", 1)[-1]: v for k, v in sd["state_dict"].items()}
        m.load_state_dict(sd, strict=False)
        if dtype is not None:
            m = m.to(dtype)
        return m.to(device)
    return _f


def _orthrus_encode(tokenizer, seqs, max_len):
    from external_models.common.orthrus_encode import encode_batch
    x, lengths = encode_batch(seqs)  # (B, L, 6), (B,)
    # Truncate to max_len
    if x.shape[1] > max_len:
        x = x[:, :max_len, :]
        lengths = lengths.clamp(max=max_len)
    # attention_mask from lengths
    B, L, _ = x.shape
    mask = (torch.arange(L).unsqueeze(0) < lengths.unsqueeze(1)).long()
    return {"x": x, "lengths": lengths, "attention_mask": mask}


def _orthrus_forward(backbone, inputs):
    # MixerModel.forward(x, channel_last=True) -> (B, L, d_model)
    return backbone(inputs["x"], channel_last=True)


# ───────────────────────────── evo2 (non-HF, embedding only) ─────────────────
EVO2_EMBED_LAYER = "blocks.28.mlp.l3"  # mid-network layer for 7B embeddings (per evo2 README)
_EVO2_CACHE = {}


def _evo2_load(model_dir):
    def _f(device="cuda:0", dtype=None):
        sys.path.insert(0, str(Path(model_dir)))
        from evo2.models import Evo2
        ckpt = os.path.join(model_dir, "evo2_7b.pt")
        ev = Evo2(model_name="evo2_7b", local_path=ckpt)
        _EVO2_CACHE["model"] = ev  # so the tokenizer fn can reuse it
        return ev  # wrapper, not nn.Module; embedding-only
    return _f


def _evo2_tok(model_dir):
    def _f():
        # tokenizer is owned by the Evo2 wrapper; reuse the loaded instance if present
        ev = _EVO2_CACHE.get("model")
        if ev is not None:
            return ev.tokenizer
        sys.path.insert(0, str(Path(model_dir)))
        from vortex.model.tokenizer import CharLevelTokenizer
        return CharLevelTokenizer(512)
    return _f


def _evo2_encode(tokenizer, seqs, max_len):
    ids = []
    for s in seqs:
        s = str(s).upper().replace("U", "T")[:max_len]
        toks = [int(x) for x in tokenizer.tokenize(s)]
        ids.append(torch.tensor(toks, dtype=torch.long))
    maxlen = max(len(t) for t in ids)
    pad_id = int(getattr(tokenizer, "pad_id", 1))
    rows = [torch.nn.functional.pad(t, (0, maxlen - len(t)), value=pad_id) for t in ids]
    return {"input_ids": torch.stack(rows)}


def _evo2_forward(backbone, inputs):
    # Evo2.forward returns (logits, embeddings_dict) when return_embeddings=True
    _, emb = backbone.forward(inputs["input_ids"], return_embeddings=True,
                              layer_names=[EVO2_EMBED_LAYER])
    return emb[EVO2_EMBED_LAYER]  # (B, L, H)


# ───────────────────────────── InvaRNA (non-HF Caduceus/Mamba) ───────────────
def _invarna_paths(invarna_root):
    """Idempotently put repo root (for `import src.utils`) and src/ (for
    `import trainmodule`, `from backbone.tokenization import ...`) on sys.path,
    with src/ ending at path[0]."""
    for p in (str(Path(invarna_root)), str(Path(invarna_root) / "src")):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


def _invarna_load(invarna_root):
    def _f(device="cuda:0", dtype=None):
        _invarna_paths(invarna_root)
        from omegaconf import OmegaConf
        from trainmodule import SequenceLightningModule
        cfg = OmegaConf.load(os.path.join(invarna_root, "config", "backbone.yaml"))
        model = SequenceLightningModule(config=cfg)
        sd = torch.load(
            os.path.join(invarna_root, "checkpoints", "backbone",
                         "model_weights0718step13500.pt"),
            map_location="cpu", weights_only=False,
        )
        model.model.load_state_dict(sd)            # strict; matches reference recipe
        backbone = model.model.invaRNA             # Caduceus base encoder (no LM head)
        backbone.config.return_dict = True
        if dtype is not None:
            backbone = backbone.to(dtype)
        return backbone.to(device)
    return _f


def _invarna_tok(invarna_root):
    def _f():
        _invarna_paths(invarna_root)
        from backbone.tokenization import InvaRNATokenizer
        return InvaRNATokenizer(model_max_length=10000)
    return _f


def _invarna_encode(tokenizer, seqs, max_len):
    """DNA alphabet (A,C,G,T,N). U->T so RNA input isn't mapped to UNK(6);
    N->pad per the reference recipe; manual right-pad + attention_mask that
    masks ALL pad ids (batch-pad AND N-derived) so masked mean pooling is correct."""
    pad_id = tokenizer.pad_token_id                    # 4
    n_id = tokenizer.convert_tokens_to_ids("N")        # 11
    rows = []
    for s in seqs:
        s = str(s).upper().replace("U", "T")
        ids = tokenizer.encode(s, add_special_tokens=False)[:max_len]
        ids = [pad_id if t == n_id else t for t in ids]
        rows.append(ids or [pad_id])
    maxlen = max(len(r) for r in rows)
    rows = [r + [pad_id] * (maxlen - len(r)) for r in rows]
    input_ids = torch.tensor(rows, dtype=torch.long)
    attention_mask = (input_ids != pad_id).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _invarna_forward(backbone, inputs):
    # Caduceus.forward does NOT accept attention_mask -> pass input_ids ONLY.
    out = backbone(input_ids=inputs["input_ids"])
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state               # (B, L, 512)
    return out[0]


# ═══════════════════════════════ REGISTRY ════════════════════════════════════
# head_mode='native' -> use the model's OWN regression/cls head for full-FT.
#   native_loader='multimolecule' -> AutoModelForSequencePrediction (rnafm)
#   native_loader='hf_auto'       -> transformers.AutoModelForSequenceClassification
# head_mode='shared' -> no native head; full-FT uses unified SharedSeqClsHead (orthrus).
def _spec_rnafm():
    d = str(MODEL_ROOT / "RNA-FM")
    return BackboneSpec("rnafm", d, 640, "threeutrlm_env", "rna",
                        _mm_load(d), _mm_tok(d), _hf_encode, _hf_last_hidden,
                        head_mode="native", native_loader="multimolecule")


def _spec_codonbert():
    d = str(MODEL_ROOT / "CodonBERT")
    return BackboneSpec("codonbert", d, 768, "agent2", "cds",
                        _codonbert_load(d), _codonbert_tok(d), _codonbert_encode, _hf_last_hidden,
                        head_mode="native", native_loader="hf_auto",
                        regions=("cds",))


def _spec_dnabert2():
    d = str(MODEL_ROOT / "DNABERT2")
    return BackboneSpec("dnabert2", d, 768, "agent2", "dna",
                        _dnabert2_load(d), _auto_tok(d), _dna_encode, _hf_last_hidden,
                        head_mode="native", native_loader="hf_auto",
                        notes="flash-attn GPU only; full-FT uses PyTorch attn path")


def _spec_mrnabert():
    d = str(MODEL_ROOT / "mRNABERT")
    return BackboneSpec("mrnabert", d, 768, "mrnabert_env", "rna",
                        _trust_remote_load(d), _auto_tok(d), _mrnabert_encode, _hf_last_hidden,
                        head_mode="native", native_loader="hf_auto",
                        notes="space-separated tokens")


def _spec_lucaone():
    d = str(MODEL_ROOT / "lucaone_hf_ckpts" / "LucaOne-gene-step36.8M")
    return BackboneSpec("lucaone", d, 2560, "lucaone", "gene",
                        _lucaone_load(d), _lucaone_tok(d), _lucaone_encode, _hf_last_hidden,
                        head_mode="native", native_loader="hf_auto",
                        notes="seq_type=gene; 2560-d", default_max_len=1280)


def _spec_orthrus():
    d = str(MODEL_ROOT / "Orthrus")
    return BackboneSpec("orthrus", d, 512, "caduceus_env", "rna",
                        _orthrus_load(d), lambda: (lambda: None)(), _orthrus_encode, _orthrus_forward,
                        head_mode="shared",
                        notes="non-HF Mamba; 6-track one-hot; full-FT uses SharedSeqClsHead")


def _spec_evo2():
    d = str(MODEL_ROOT / "evo2-main")
    return BackboneSpec("evo2", d, 4096, "mamballm", "dna",
                        _evo2_load(d), _evo2_tok(d), _evo2_encode, _evo2_forward,
                        supports_full_ft=False, head_mode="shared",
                        notes="non-HF StripedHyena; embedding via run_embedding.py; "
                              "full-FT via sft/finetune_evo2.py (custom loop, vortex 7-GPU shard, NOT finetune_unified.py)")


def _spec_invarna():
    d = INVARNA_ROOT
    return BackboneSpec("invarna", d, 512, "mamballm", "rna",
                        _invarna_load(d), _invarna_tok(d), _invarna_encode, _invarna_forward,
                        supports_full_ft=False, head_mode="shared",
                        notes="non-HF Caduceus/Mamba (rcps d_model 256->512); GPU only "
                              "(mamba_ssm/Triton fused_add_norm); embedding only; U->T, "
                              "N->pad; repo is sibling of llm_models/ (env INVARNA_ROOT)")


_BUILDERS = {
    "rnafm": _spec_rnafm,
    "codonbert": _spec_codonbert,
    "dnabert2": _spec_dnabert2,
    "mrnabert": _spec_mrnabert,
    "lucaone": _spec_lucaone,
    "orthrus": _spec_orthrus,
    "evo2": _spec_evo2,
    "invarna": _spec_invarna,
}

ALL_KEYS = list(_BUILDERS.keys())
FULL_FT_KEYS = [k for k in ALL_KEYS if k not in ("evo2", "invarna")]


def get_spec(key: str) -> BackboneSpec:
    if key not in _BUILDERS:
        raise ValueError(f"Unknown backbone '{key}'. Available: {ALL_KEYS}")
    return _BUILDERS[key]()


# ───────────────── native seq-cls/regression head loaders (full-FT) ──────────
def load_native_seqcls(spec, num_labels=1, problem_type="regression", device="cuda:0"):
    """Load the backbone WITH its own native regression/classification head.

    Returns a standard HF model whose forward(**inputs, labels=) -> output.loss/logits.
    Used by finetune_unified.py when spec.head_mode == 'native'.
    """
    if spec.native_loader == "multimolecule":
        # RNA-FM: multimolecule SequencePrediction head
        from multimolecule import AutoModelForSequencePrediction, AutoConfig as MMAutoConfig
        cfg = MMAutoConfig.from_pretrained(spec.model_dir)
        cfg.num_labels = num_labels
        cfg.problem_type = problem_type
        if getattr(cfg, "head", None) is not None:
            try:
                cfg.head.num_labels = num_labels
                cfg.head.problem_type = problem_type
            except Exception:
                pass
        m = AutoModelForSequencePrediction.from_pretrained(
            spec.model_dir, config=cfg, local_files_only=True)
        return m.to(device)

    # hf_auto: codonbert / dnabert2 / mrnabert / lucaone
    from transformers import AutoModelForSequenceClassification, AutoConfig
    extra = {}
    if spec.key == "lucaone":
        _ensure_lucaone_importable()
    if spec.key in ("dnabert2", "mrnabert", "lucaone"):
        extra["trust_remote_code"] = True

    cfg = AutoConfig.from_pretrained(spec.model_dir, **extra)
    cfg.num_labels = num_labels
    cfg.problem_type = problem_type
    # DNABERT2: force PyTorch attention path (Triton kernel breaks on triton>=3)
    if spec.key == "dnabert2" and getattr(cfg, "attention_probs_dropout_prob", 0.0) == 0.0:
        cfg.attention_probs_dropout_prob = 0.1
    # LucaOne: its head reads classifier_* fields from config
    if spec.key == "lucaone":
        cfg.task_level = "seq_level"
        cfg.task_type = problem_type if problem_type in (
            "regression", "binary_class", "multi_class", "multi_label") else "regression"
        cfg.classifier_num_labels = num_labels

    m = AutoModelForSequenceClassification.from_pretrained(spec.model_dir, config=cfg, **extra)
    return m.to(device)
