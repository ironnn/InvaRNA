"""
evo2 (StripedHyena 7B) full-parameter fine-tuning.

Why a dedicated module (not unified/model.py + HF Trainer):
  - evo2 is NOT a HuggingFace model. vortex's StripedHyena does its OWN pipeline
    parallelism, auto-sharding layer blocks across all visible GPUs
    (block_idx_to_device). HF Trainer would call model.to(device) / wrap in
    DataParallel and collapse that sharding -> OOM. So we use a custom loop.
  - The Evo2 wrapper's forward() has torch.no_grad() + output.detach() (inference
    only). We use the RAW StripedHyena (Evo2(...).model) and a forward hook on
    `norm` to tap the pre-unembed hidden state WITH gradient.
  - vortex loads weights under torch.inference_mode() (vortex/model/utils.py:108),
    permanently tainting them as inference tensors. We re-materialize every param/
    buffer outside inference_mode into fresh nn.Parameter so autograd works.

Head: evo2 has no native regression head, so it uses the shared SharedSeqClsHead
(consistent with the head policy: native if present, else shared).

Sharding: the backbone is split across GPUs by vortex; the head + loss live on the
block-0 device (where `norm` output lands, per model.py:715). This is pipeline
parallelism within a single process — NOT DDP, do NOT launch with torchrun.
"""
import os
import sys
import torch
import torch.nn as nn

BASELINES_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BASELINES_ROOT)
sys.path.insert(0, REPO_ROOT)
from external_models.common.shared_head import SharedSeqClsHead


def _evo2_dir():
    asset_root = os.environ.get(
        "INVARNA_EXTERNAL_ASSET_ROOT",
        os.path.join(REPO_ROOT, "assets", "checkpoints", "external", "pretrained"),
    )
    return os.path.join(asset_root, "models", "evo2-main")


def load_evo2_trainable():
    """Load raw StripedHyena, re-materialize params as autograd-capable, return (model, tokenizer)."""
    sys.path.insert(0, _evo2_dir())
    from evo2.models import Evo2
    ckpt = os.path.join(_evo2_dir(), "evo2_7b.pt")
    ev = Evo2(model_name="evo2_7b", local_path=ckpt)
    model, tok = ev.model, ev.tokenizer

    # un-taint inference tensors: clone every param/buffer outside inference_mode
    def _rematerialize(module):
        for name, p in list(module._parameters.items()):
            if p is not None:
                module._parameters[name] = nn.Parameter(p.detach().clone(), requires_grad=True)
        for name, b in list(module._buffers.items()):
            if b is not None:
                module._buffers[name] = b.detach().clone()

    with torch.inference_mode(False):
        for m in model.modules():
            _rematerialize(m)
    return model, tok


class Evo2ForSeqCls(nn.Module):
    """Raw StripedHyena backbone + shared head, with gradient-capable hidden-state tap."""
    def __init__(self, num_labels=1, task_type="regression", pooling="mean"):
        super().__init__()
        self.backbone, self.tokenizer = load_evo2_trainable()
        self.hidden_size = self.backbone.config.hidden_size           # 4096
        self.dev0 = self.backbone.block_idx_to_device[0]              # where norm output lands
        self.task_type = task_type
        self.head = SharedSeqClsHead(
            hidden_size=self.hidden_size, num_labels=num_labels,
            task_type=task_type, pooling=pooling,
        ).to(self.dev0).to(torch.bfloat16)

        # forward hook on norm captures pre-unembed hidden state (B, L, H) WITH grad
        self._captured = {}
        def _hook(_, __, output):
            self._captured["h"] = output[0] if isinstance(output, tuple) else output
        self.backbone.norm.register_forward_hook(_hook)

    def encode(self, seqs, max_len=1024):
        ids = []
        for s in seqs:
            s = str(s).upper().replace("U", "T")[:max_len]
            ids.append(torch.tensor([int(x) for x in self.tokenizer.tokenize(s)], dtype=torch.long))
        L = max(len(t) for t in ids)
        pad = int(self.tokenizer.pad_id)
        batch = torch.stack([torch.nn.functional.pad(t, (0, L - len(t)), value=pad) for t in ids])
        attn = torch.stack([
            torch.cat([torch.ones(len(t)), torch.zeros(L - len(t))]) for t in ids
        ]).long()
        return batch.to(self.dev0), attn.to(self.dev0)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # raw forward (grad enabled); norm hook fills self._captured["h"]
        self.backbone(input_ids)
        h = self._captured["h"].to(self.dev0)              # (B, L, H), grad intact
        am = attention_mask.to(self.dev0) if attention_mask is not None else None
        lab = labels.to(self.dev0) if labels is not None else None
        logits, loss = self.head(h, attention_mask=am, labels=lab)
        return logits, loss

    def trainable_param_report(self):
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        return n_train, n_total
