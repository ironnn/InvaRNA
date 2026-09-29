import torch
import torch.nn as nn
import torch.nn.functional as F

class IdentityExpert(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class CNNExpert(nn.Module):
    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size

        if kernel_size % 2 == 1:
            self.prepad = None
            padding = kernel_size // 2
        else:
            left = kernel_size // 2 - 1
            right = kernel_size // 2
            self.prepad = nn.ConstantPad1d((left, right), 0)
            padding = 0


        self.net = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=kernel_size, padding=padding, groups=2),
            nn.SiLU(),
            nn.Conv1d(2 * d_model, d_model, kernel_size=1, groups=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, D, L)
        if self.prepad is not None:
            x = self.prepad(x)
        x = self.net(x)
        return x.transpose(1, 2)  # (B, L, D)

class CNNMoELayer(nn.Module):
    def __init__(self, d_model: int, num_experts: int = 5, region_dim: int = 16, utr_length: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.region_dim = region_dim
        self.utr_length = utr_length

        # region embedding: 0 = 5'UTR, 1 = CDS+3'UTR
        self.region_embedding = nn.Embedding(2, region_dim)


        self.gate_fwd = nn.Sequential(
            nn.Linear(d_model + region_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_experts)
        )
        self.gate_rc = nn.Sequential(
            nn.Linear(d_model + region_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_experts)
        )

        # experts: residual + CNN with multiple kernels
        kernel_sizes = [3, 9, 28]
        self.experts = nn.ModuleList([
            IdentityExpert(),  # Expert 0: residual
            *[CNNExpert(d_model * 2, k) for k in kernel_sizes]  # experts operate on full dim
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, 2 * d_model), where first half is fwd, second half is rc
        return: (B, L, 2 * d_model), gate_logits_fwd: (B, L, E)
        """
        B, L, D = x.shape
        d_model = D // 2
        x_fwd, x_rc = x[..., :d_model], x[..., d_model:]

        # region embedding
        region_id = torch.zeros(B, L, dtype=torch.long, device=x.device)
        region_id[:, self.utr_length:] = 1
        region_emb = self.region_embedding(region_id)  # (B, L, region_dim)

        # gating
        gate_input_fwd = torch.cat([x_fwd, region_emb], dim=-1)  # (B, L, d_model + region_dim)
        gate_input_rc  = torch.cat([x_rc, region_emb], dim=-1)
        gate_logits_fwd = self.gate_fwd(gate_input_fwd)  # (B, L, E)
        gate_logits_rc  = self.gate_rc(gate_input_rc)
        gate_fwd = F.softmax(gate_logits_fwd, dim=-1).unsqueeze(-1)  # (B, L, E, 1)
        gate_rc  = F.softmax(gate_logits_rc,  dim=-1).unsqueeze(-1)

        # shared expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)  # (B, L, E, 2*d_model)
        expert_fwd = expert_outputs[..., :d_model]  # (B, L, E, d_model)
        expert_rc  = expert_outputs[..., d_model:]  # (B, L, E, d_model)

        out_fwd = (expert_fwd * gate_fwd).sum(dim=2)  # (B, L, d_model)
        out_rc  = (expert_rc  * gate_rc ).sum(dim=2)

        output = torch.cat([out_fwd, out_rc], dim=-1)  # (B, L, 2*d_model)

        return output, gate_logits_fwd
