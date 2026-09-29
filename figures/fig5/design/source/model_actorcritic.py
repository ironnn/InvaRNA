# rl_system/model_actorcritic.py
import torch
import torch.nn as nn
from typing import Tuple

class ActorCritic(nn.Module):
    """
    简洁的一维序列网络：
    - Embedding(5→d)：支持 A/C/G/T/N 五种碱基
    - 1D CNN 提取局部特征
    - 全局池化 + MLP，输出策略 logits 与值函数 value

    输入：
        x_idx: (B, L) LongTensor
    输出：
        logits: (B, action_dim)
        value : (B, 1)
    """
    def __init__(self, seq_len: int, action_dim: int, d_model: int = 64):
        super().__init__()
        self.seq_len = seq_len
        self.action_dim = action_dim

        # A/C/G/T/N → 向量
        self.emb = nn.Embedding(5, d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)

        hidden = 2 * d_model
        self.head_policy = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        self.head_value = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_idx: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_idx: (B, L) LongTensor
        return:
            logits: (B, action_dim)
            value : (B, 1)
        """
        x = self.emb(x_idx)          # (B, L, d)
        x = x.transpose(1, 2)        # (B, d, L)
        x = self.conv(x)             # (B, d, L)
        x = self.act(x)
        x = self.pool(x).squeeze(-1) # (B, d)

        logits = self.head_policy(x)
        value = self.head_value(x)
        return logits, value
