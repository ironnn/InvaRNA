#stage2_definitions.py

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics.functional import r2_score

# ==========================================
# Mamba + RC-Safe CNN Head 模型定义
# ==========================================
class MambaRCHeadStage2(LightningModule):
    def __init__(self, pretrained_backbone, config, head_lr=1e-3, backbone_lr=5e-6):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_backbone'])
        self.config = config


        self.backbone = pretrained_backbone


        d_model = 512
        self.rc_cnn_head = nn.Sequential(

            nn.Conv1d(d_model, 256, kernel_size=5, padding=2, groups=2),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.MaxPool1d(2),

            # Block 2: L/2 -> L/4
            nn.Conv1d(256, 128, kernel_size=5, padding=2, groups=2),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.MaxPool1d(2),

            # Block 3: L/4 -> L/8
            nn.Conv1d(128, 128, kernel_size=5, padding=2, groups=2),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.MaxPool1d(2),


            nn.AdaptiveAvgPool1d(64)
        )


        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 64, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask=None):
        # 1. Backbone
        hidden_states, *_ = self.backbone(
            input_ids,
            output_hidden_states=False,
            return_dict=False
        )

        if attention_mask is not None:
             mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
             hidden_states = hidden_states * mask

        # 2. Permute to (B, D, L) for CNN
        x = hidden_states.permute(0, 2, 1)

        # 3. CNN Head + Regressor
        x = self.rc_cnn_head(x)
        return self.regressor(x)