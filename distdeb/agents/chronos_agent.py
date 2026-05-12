"""Chronos-Bolt agent wrapper, with preprocessing-mode specialists.

The vanilla Chronos-Bolt is a zero-shot foundation TS forecaster. By feeding
it preprocessed views of the same series we induce *complementary* forecasts
from the same backbone — the specialists outlined in DESIGN.md §4. They have
similar headline CRPS (same model) but err on different regimes, which is
what panel diversification requires.

Supported preprocessors (each implemented as `_forecast_<name>`):
  - identity: no preprocessing (the canonical foundation forecast).
  - detrend:  remove per-series linear trend, forecast residual, add back.
  - diff:     forecast first-differences, integrate by cumsum + last value.

Reference: Ansari et al., "Chronos: Learning the Language of Time Series",
2024. Repo: https://github.com/amazon-science/chronos-forecasting
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Agent, Forecast


_VALID_PREPROCESSORS = {"identity", "detrend", "diff"}


class ChronosBoltAgent(Agent):
    """Wraps amazon/chronos-bolt-* with our Agent interface.

    Specialist variants are realized by setting `preprocessor`:
      ChronosBoltAgent(preprocessor="detrend")  # trend specialist
      ChronosBoltAgent(preprocessor="diff")     # difference specialist
    """

    nominal_cost = 1.0

    def __init__(
        self,
        model_id: str = "amazon/chronos-bolt-base",
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        name_suffix: str = "",
        preprocessor: str = "identity",
    ):
        import torch
        from chronos import BaseChronosPipeline

        assert preprocessor in _VALID_PREPROCESSORS, (
            f"preprocessor must be one of {_VALID_PREPROCESSORS}, got {preprocessor!r}"
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_id, device_map=device, torch_dtype=getattr(torch, dtype)
        )
        self.device = device
        self.model_id = model_id
        self.preprocessor = preprocessor
        suffix = name_suffix or ("_" + preprocessor if preprocessor != "identity" else "")
        self.name = f"chronos_bolt{suffix}"

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
            )
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
        """Batched univariate forecast. histories: (N, L) -> (N, Q, H)."""
        histories = np.asarray(histories, dtype=np.float32)
        assert histories.ndim == 2, f"expected (N, L), got {histories.shape}"
        return getattr(self, f"_forecast_{self.preprocessor}")(histories, horizon, np.asarray(levels), batch_size)

    def _forecast_uni(self, x: np.ndarray, horizon: int, levels: np.ndarray) -> np.ndarray:
        return self.forecast_batch(x[None, :], horizon, levels, batch_size=1)[0]

    # ----- preprocessor implementations -----

    def _forecast_identity(self, histories: np.ndarray, horizon: int, levels: np.ndarray, batch_size: int) -> np.ndarray:
        return self._call_pipeline(histories, horizon, levels, batch_size)

    def _forecast_detrend(self, histories: np.ndarray, horizon: int, levels: np.ndarray, batch_size: int) -> np.ndarray:
        N, L = histories.shape
        t = np.arange(L, dtype=np.float32)
        # Per-series least-squares linear fit.
        slopes, intercepts = np.polyfit(t, histories.T, 1)  # each (N,)
        trend_hist = slopes[:, None] * t[None, :] + intercepts[:, None]
        resid = histories - trend_hist
        quants_resid = self._call_pipeline(resid, horizon, levels, batch_size)  # (N, Q, H)
        future_t = np.arange(L, L + horizon, dtype=np.float32)
        trend_future = slopes[:, None] * future_t[None, :] + intercepts[:, None]  # (N, H)
        return (quants_resid + trend_future[:, None, :]).astype(np.float32)

    def _forecast_diff(self, histories: np.ndarray, horizon: int, levels: np.ndarray, batch_size: int) -> np.ndarray:
        # First-difference, forecast in diff space, integrate via cumsum + last value.
        diffs = np.diff(histories, axis=1)
        last = histories[:, -1]
        quants_diff = self._call_pipeline(diffs, horizon, levels, batch_size)  # (N, Q, H)
        # Cumsum across horizon for each quantile (treating each quantile path independently).
        # This is an approximation: the true quantile of an integrated path is not just
        # the cumsum of marginal quantiles. For panel-member purposes it's fine.
        return (np.cumsum(quants_diff, axis=-1) + last[:, None, None]).astype(np.float32)

    def _call_pipeline(self, histories: np.ndarray, horizon: int, levels: np.ndarray, batch_size: int) -> np.ndarray:
        import torch

        N, _ = histories.shape
        Q = len(levels)
        out = np.empty((N, Q, horizon), dtype=np.float32)
        for i in range(0, N, batch_size):
            batch = torch.tensor(histories[i : i + batch_size], dtype=torch.float32)
            # Chronos >=1.5: `inputs` (was `context`). Returns CPU fp32 tensors.
            quantiles, _mean = self.pipeline.predict_quantiles(
                inputs=batch,
                prediction_length=horizon,
                quantile_levels=list(map(float, levels)),
            )
            arr = quantiles.detach().to("cpu").float().numpy()  # (B, H, Q)
            out[i : i + arr.shape[0]] = np.transpose(arr, (0, 2, 1))
        return out
