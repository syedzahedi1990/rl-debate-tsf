"""Convolutional encoder for raw lookback windows.

The Path C diagnostic showed that hand-crafted summary statistics
(10-dim feature vector) cap the policy's ability to learn dataset- and
regime-conditional routing. A small Conv1D encoder over the raw lookback
should extract richer features without the user pre-specifying them.

Architecture: 3 strided 1D convolutions + adaptive avg pool + linear
projection. ~3k params, fast on a single A100.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Conv1DEncoder(nn.Module):
    def __init__(self, output_dim: int = 32, in_channels: int = 1, hidden: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) or (B, 1, L) -> (B, output_dim)."""
        if x.ndim == 2:
            x = x.unsqueeze(1)
        h = self.conv(x).squeeze(-1)  # (B, hidden)
        return self.proj(h)
