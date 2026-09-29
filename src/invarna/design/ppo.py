"""Checkpoint-compatible actor-critic network used by the design workflow.

The recovered repository does not yet contain a validated manuscript PPO training
launcher. This module intentionally exposes only the network and sampling operation.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, sequence_length, action_dim, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(4, d_model)
        self.position = nn.Parameter(torch.zeros(1, sequence_length, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.actor = nn.Linear(d_model, action_dim)
        self.critic = nn.Linear(d_model, 1)

    def forward(self, sequence):
        if sequence.ndim == 1:
            sequence = sequence.unsqueeze(0)
        state = self.encoder(self.embedding(sequence) + self.position).mean(dim=1)
        return self.actor(state), self.critic(state).squeeze(-1)

    @torch.no_grad()
    def act(self, sequence):
        logits, value = self(sequence)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        return action, distribution.log_prob(action), value
