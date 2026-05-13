"""Small MLP (optionally + Conv1D) policy + value head.

Default: pure MLP over a flat state (~50 dims).

If the env is configured with `include_history`, the state contains a raw
lookback window contiguously at `[history_offset : history_offset+history_dim]`.
Passing `history_offset` and `history_dim` here switches on a Conv1D encoder
over that slice; the rest of the state is fed to the trunk unchanged. This
adds ~3k parameters and gives the policy a richer regime representation than
hand-crafted summary stats alone.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .encoders import Conv1DEncoder


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: int = 128,
        history_offset: Optional[int] = None,
        history_dim: Optional[int] = None,
        conv_output_dim: int = 32,
    ):
        super().__init__()
        self.use_conv = history_offset is not None and history_dim is not None
        if self.use_conv:
            self.history_offset = int(history_offset)
            self.history_dim = int(history_dim)
            self.encoder = Conv1DEncoder(output_dim=conv_output_dim)
            trunk_in = state_dim - self.history_dim + conv_output_dim
        else:
            trunk_in = state_dim

        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)
        for layer in (self.policy_head, self.value_head):
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_conv:
            before = x[:, : self.history_offset]
            history = x[:, self.history_offset : self.history_offset + self.history_dim]
            after = x[:, self.history_offset + self.history_dim :]
            enc = self.encoder(history)
            x = torch.cat([before, enc, after], dim=-1)
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)
