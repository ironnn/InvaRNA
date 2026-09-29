"""
Shared sequence-classification / regression head.

Backbone-agnostic: takes a backbone's last_hidden_state (B, L, H) + attention_mask
and produces logits + loss. This is the SINGLE head shared by every backbone in the
unified full-parameter fine-tuning harness.

Distilled from LucaOne's LucaGPLMForSequenceClassification
(lucaone_hf_pkg/src/lucaone/modeling_lucaone.py:1142-1248), but with no dependency
on any specific backbone or transformers version (pure torch).

Task-type contract (same as LucaOne):
  regression   -> MSE (or L1),     num_labels = 1,        float labels
  binary_class -> BCEWithLogits,   num_labels = 1,        float 0/1 labels
  multi_class  -> CrossEntropy,    num_labels = #classes, long labels
  multi_label  -> BCEWithLogits,   num_labels = #labels,  float 0/1 vectors
"""
import torch
import torch.nn as nn


class SharedSeqClsHead(nn.Module):
    """Pooling + linear classifier + task-aware loss. Shared across all backbones."""

    def __init__(
        self,
        hidden_size: int,
        num_labels: int = 1,
        task_type: str = "regression",
        pooling: str = "mean",
        dropout_prob: float = 0.1,
        loss_type: str = "mse",          # for regression: 'mse' | 'mae'
        pos_weight: float = None,        # for binary/multi_label
        class_weight=None,               # for multi_class
    ):
        super().__init__()
        assert task_type in ("regression", "binary_class", "multi_class", "multi_label")
        assert pooling in ("mean", "cls", "max")

        # task-type / num_labels contract
        if task_type in ("regression", "binary_class"):
            num_labels = 1

        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.task_type = task_type
        self.pooling = pooling

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

        if task_type == "multi_class":
            w = torch.tensor(class_weight, dtype=torch.float32) if class_weight is not None else None
            self.loss_fct = nn.CrossEntropyLoss(weight=w, reduction="mean")
        elif task_type == "binary_class":
            pw = torch.tensor([pos_weight], dtype=torch.float32) if pos_weight is not None else None
            self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="mean")
        elif task_type == "multi_label":
            pw = (torch.tensor([pos_weight] * num_labels, dtype=torch.float32)
                  if pos_weight is not None else None)
            self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="mean")
        else:  # regression
            self.loss_fct = nn.L1Loss(reduction="mean") if loss_type == "mae" else nn.MSELoss(reduction="mean")

    def _pool(self, hidden_states, attention_mask):
        """(B, L, H) + (B, L) -> (B, H)."""
        if self.pooling == "cls":
            return hidden_states[:, 0, :]

        if attention_mask is None:
            if self.pooling == "max":
                return hidden_states.max(dim=1)[0]
            return hidden_states.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # (B, L, 1)
        if self.pooling == "max":
            neg = torch.finfo(hidden_states.dtype).min
            masked = hidden_states.masked_fill(mask == 0, neg)
            return masked.max(dim=1)[0]
        # masked mean
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(self, hidden_states, attention_mask=None, labels=None):
        pooled = self._pool(hidden_states, attention_mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (B, num_labels)

        loss = None
        if labels is not None:
            # Compute loss in fp32 for numerical stability (logits may be bf16).
            logits_f = logits.float()
            if self.task_type == "multi_class":
                loss = self.loss_fct(logits_f.view(-1, self.num_labels), labels.view(-1).long())
            elif self.task_type == "binary_class":
                loss = self.loss_fct(logits_f.view(-1), labels.view(-1).float())
            elif self.task_type == "regression":
                loss = self.loss_fct(logits_f.view(-1), labels.view(-1).float())
            elif self.task_type == "multi_label":
                loss = self.loss_fct(logits_f.view(-1, self.num_labels), labels.view(-1, self.num_labels).float())

        return logits, loss
