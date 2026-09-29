# rl_system/encoding.py
import torch

# -----------------------------
# 基础字典定义
# -----------------------------
BASE2IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
IDX2BASE = {v: k for k, v in BASE2IDX.items()}

# -----------------------------
# 序列 ↔ 张量转换
# -----------------------------
def encode_seq(seq: str) -> torch.LongTensor:
    """将 'ACGTN' 序列编码为 LongTensor"""
    seq = seq.upper()
    return torch.tensor([BASE2IDX.get(b, 4) for b in seq], dtype=torch.long)

def decode_seq(idx_arr: torch.LongTensor) -> str:
    """将 LongTensor 还原为 'ACGTN' 序列"""
    return "".join(IDX2BASE.get(int(i), "N") for i in idx_arr)
