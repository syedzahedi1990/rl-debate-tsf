"""Small MLP policy + value head.

Inputs: flattened state from RefinementEnv (state_dim ~50).
Outputs: logits over n_actions + scalar value.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)
        # Mild init for value head to avoid spurious early bias.
        for layer in (self.policy_head, self.value_head):
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)
