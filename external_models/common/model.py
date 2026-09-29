"""
UnifiedBackboneForSeqCls — any backbone's base encoder + the SINGLE SharedSeqClsHead.

Used by finetune_unified.py for full-parameter fine-tuning: the entire backbone PLUS
the shared head receive gradients. Works with the HF Trainer (forward returns a dict
with 'loss' and 'logits').

The backbone-specific quirks (tuple outputs, seq_type, 6-track input) are absorbed by
the BackboneSpec.forward_hidden / encode callables in unified/backbones.py — this
wrapper stays generic.
"""
import torch
import torch.nn as nn

from external_models.common.shared_head import SharedSeqClsHead
from external_models.common.backbones import get_spec, load_native_seqcls


# task_type (our naming) -> HF problem_type
TASK_TO_PROBLEM_TYPE = {
    "regression": "regression",
    "binary_class": "regression",         # 1-logit BCE-ish; HF uses regression slot w/ num_labels=1 -> MSE
    "multi_class": "single_label_classification",
    "multi_label": "multi_label_classification",
}


class NativeHeadModel(nn.Module):
    """Wrap a model's OWN HF/multimolecule seq-cls head so it matches the Trainer
    forward contract used here. Backbone-specific encode quirks come from the registry.

    Used when spec.head_mode == 'native' (6 of the backbones ship their own head).
    """
    def __init__(self, backbone_key, num_labels=1, task_type="regression",
                 device="cuda:0"):
        super().__init__()
        self.spec = get_spec(backbone_key)
        self.backbone_key = backbone_key
        self.task_type = task_type
        problem_type = TASK_TO_PROBLEM_TYPE.get(task_type, "regression")
        self.model = load_native_seqcls(
            self.spec, num_labels=num_labels, problem_type=problem_type, device=device)
        self._device = device

    def gradient_checkpointing_enable(self, **kwargs):
        m = getattr(self.model, "gradient_checkpointing_enable", None)
        if callable(m):
            # use_reentrant=False is required for grad-ckpt + DDP compatibility
            try:
                m(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                try:
                    m()
                except (ValueError, NotImplementedError) as e:
                    print(f"[warn] {self.backbone_key}: gradient_checkpointing unsupported ({e}); skipping")
            except (ValueError, NotImplementedError) as e:
                print(f"[warn] {self.backbone_key}: gradient_checkpointing unsupported ({e}); skipping")

    @property
    def config(self):
        return getattr(self.model, "config", None)

    def forward(self, labels=None, **inputs):
        inputs.pop("num_items_in_batch", None)
        out = self.model(**inputs, labels=labels)
        # normalize to dict
        if hasattr(out, "loss"):
            res = {"logits": out.logits}
            if out.loss is not None:
                res["loss"] = out.loss
            return res
        return out

    def trainable_param_report(self):
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        return n_train, n_total


class UnifiedBackboneForSeqCls(nn.Module):
    def __init__(
        self,
        backbone_key: str,
        num_labels: int = 1,
        task_type: str = "regression",
        pooling: str = "mean",
        device: str = "cuda:0",
        dtype=None,
        loss_type: str = "mse",
        pos_weight: float = None,
        class_weight=None,
    ):
        super().__init__()
        self.spec = get_spec(backbone_key)
        self.backbone_key = backbone_key
        self.task_type = task_type

        self.backbone = self.spec.load_backbone(device=device, dtype=dtype)
        self.head = SharedSeqClsHead(
            hidden_size=self.spec.hidden_dim,
            num_labels=num_labels,
            task_type=task_type,
            pooling=pooling,
            loss_type=loss_type,
            pos_weight=pos_weight,
            class_weight=class_weight,
        ).to(device)
        # Keep head dtype consistent with the backbone (avoids bf16/fp32 matmul errors).
        if dtype is not None:
            self.head = self.head.to(dtype)
        self._device = device
        self._dtype = dtype

    # --- HF Trainer compatibility ---------------------------------------------
    def gradient_checkpointing_enable(self, **kwargs):
        """Best-effort: some custom backbones (DNABERT2) don't support it."""
        m = getattr(self.backbone, "gradient_checkpointing_enable", None)
        if callable(m):
            # use_reentrant=False is required for grad-ckpt + DDP compatibility
            try:
                m(gradient_checkpointing_kwargs={"use_reentrant": False})
                return
            except TypeError:
                try:
                    m()
                    return
                except (ValueError, NotImplementedError) as e:
                    print(f"[warn] {self.backbone_key}: gradient_checkpointing unsupported ({e}); skipping")
                    return
            except (ValueError, NotImplementedError) as e:
                print(f"[warn] {self.backbone_key}: gradient_checkpointing unsupported ({e}); skipping")
                return
        if hasattr(self.backbone, "gradient_checkpointing"):
            self.backbone.gradient_checkpointing = True

    @property
    def config(self):
        return getattr(self.backbone, "config", None)

    def forward(self, labels=None, **inputs):
        # Drop kwargs the HF Trainer injects that backbones don't accept.
        inputs.pop("num_items_in_batch", None)

        # attention_mask is used by the head's masked pooling; keep a copy.
        attention_mask = inputs.get("attention_mask", None)

        # Orthrus carries non-tensor-model kwargs ('x','lengths'); pass through as-is.
        hidden = self.spec.forward_hidden(self.backbone, inputs)  # (B, L, H)

        logits, loss = self.head(hidden, attention_mask=attention_mask, labels=labels)
        out = {"logits": logits}
        if loss is not None:
            out["loss"] = loss
        return out

    def trainable_param_report(self):
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        return n_train, n_total


def build_finetune_model(backbone_key, num_labels=1, task_type="regression",
                         pooling="mean", device="cuda:0"):
    """Factory: native head when the backbone ships one, else the shared head.

    Policy (per user): if a model has its OWN regression/cls head, use it; otherwise
    (orthrus) use SharedSeqClsHead.
    """
    spec = get_spec(backbone_key)
    if spec.head_mode == "native":
        print(f"[head] {backbone_key}: using NATIVE head ({spec.native_loader})")
        return NativeHeadModel(backbone_key, num_labels=num_labels,
                               task_type=task_type, device=device)
    print(f"[head] {backbone_key}: using SHARED SharedSeqClsHead (no native head)")
    return UnifiedBackboneForSeqCls(backbone_key, num_labels=num_labels,
                                    task_type=task_type, pooling=pooling,
                                    device=device, dtype=None)
