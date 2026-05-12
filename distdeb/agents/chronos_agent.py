"""Chronos-Bolt agent wrapper.

Chronos-Bolt is a zero-shot foundation TS forecaster (Amazon) with patch-based
quantile heads. We use it as a strong baseline panel member.

Reference:
  Ansari et al., "Chronos: Learning the Language of Time Series", 2024.
  Repo: https://github.com/amazon-science/chronos-forecasting
  Bolt blog: https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Agent, Forecast


class ChronosBoltAgent(Agent):
    """Wraps amazon/chronos-bolt-* checkpoints with our `Agent` interface.

    Supports batched forecasting for efficiency — important since we evaluate
    thousands of test windows.
    """

    nominal_cost = 1.0  # one model forward; calibrated relative to ARIMA's ~0.01

    def __init__(
        self,
        model_id: str = "amazon/chronos-bolt-base",
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        name_suffix: str = "",
    ):
        import torch
        from chronos import BaseChronosPipeline

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = getattr(torch, dtype)
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch_dtype,
        )
        self.device = device
        self.name = f"chronos_bolt{name_suffix}"
        self.model_id = model_id

    def forecast(
        self,
        history: np.ndarray,
        horizon: int,
        levels: np.ndarray,
        context: Optional[dict] = None,
    ) -> Forecast:
        history = np.asarray(history)
        if history.ndim == 2:
            qs = np.stack(
                [self._forecast_uni(history[:, v], horizon, levels) for v in range(history.shape[1])],
                axis=-1,
            )  # (Q, H, V)
        else:
            qs = self._forecast_uni(history, horizon, levels)
        return Forecast(quantiles=qs, levels=np.asarray(levels), agent_name=self.name, cost=self.nominal_cost)

    def forecast_batch(
        self,
        histories: np.ndarray,
        horizon: int,
        levels: np.ndarray,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Batched univariate forecast.

        histories: (N, L). Returns quantiles shape (N, Q, H).
        """
        import torch

        histories = np.asarray(histories)
        assert histories.ndim == 2, f"expected (N, L), got shape {histories.shape}"
        N, _ = histories.shape
        Q = len(levels)
        out = np.empty((N, Q, horizon), dtype=np.float32)
        for i in range(0, N, batch_size):
            batch = torch.tensor(histories[i : i + batch_size], dtype=torch.float32)
            # Chronos >=1.5 renamed `context` -> `inputs` and returns CPU fp32 tensors.
            quantiles, _mean = self.pipeline.predict_quantiles(
                inputs=batch,
                prediction_length=horizon,
                quantile_levels=list(map(float, levels)),
            )
            # quantiles: (B, H, Q); we want (B, Q, H)
            arr = quantiles.detach().to("cpu").float().numpy()
            out[i : i + arr.shape[0]] = np.transpose(arr, (0, 2, 1))
        return out

    def _forecast_uni(self, x: np.ndarray, horizon: int, levels: np.ndarray) -> np.ndarray:
        return self.forecast_batch(x[None, :], horizon, levels, batch_size=1)[0]
