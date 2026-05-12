from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Forecast:
    """A quantile forecast over a horizon.

    quantiles: shape (Q, H) or (Q, H, V); Q quantile levels, H horizon, optional V variates.
    levels: shape (Q,); quantile levels in [0, 1]; must be sorted ascending.
    """

    quantiles: np.ndarray
    levels: np.ndarray
    agent_name: str = ""
    cost: float = 0.0

    def __post_init__(self):
        self.quantiles = np.asarray(self.quantiles)
        self.levels = np.asarray(self.levels)
        assert self.quantiles.shape[0] == self.levels.shape[0], (
            f"quantiles axis 0 ({self.quantiles.shape[0]}) must match levels ({self.levels.shape[0]})"
        )
        assert np.all(np.diff(self.levels) > 0), "levels must be strictly ascending"

    @property
    def median(self) -> np.ndarray:
        idx = int(np.argmin(np.abs(self.levels - 0.5)))
        return self.quantiles[idx]

    def interval_width(self, alpha: float = 0.8) -> np.ndarray:
        """Width of the central alpha-coverage interval."""
        lo_level = (1 - alpha) / 2
        hi_level = 1 - lo_level
        lo = int(np.argmin(np.abs(self.levels - lo_level)))
        hi = int(np.argmin(np.abs(self.levels - hi_level)))
        return self.quantiles[hi] - self.quantiles[lo]


class Agent(ABC):
    """Uniform forecasting agent interface.

    Implementations:
      - return a `Forecast` with the requested quantile levels
      - set `name` and `nominal_cost` so the orchestrator can route cost-aware
      - may use `context` (prior forecasts, regime hints) when supplied
    """

    name: str = "base"
    nominal_cost: float = 1.0

    @abstractmethod
    def forecast(
        self,
        history: np.ndarray,
        horizon: int,
        levels: np.ndarray,
        context: Optional[dict] = None,
    ) -> Forecast:
        """Produce a quantile forecast.

        history: shape (L,) for univariate or (L, V) for multivariate.
        horizon: number of future steps to predict.
        levels: 1-d array of target quantile levels in (0, 1).
        context: optional dict — e.g. {"prior_forecasts": [...], "regime": {...}}.
        """
        ...
