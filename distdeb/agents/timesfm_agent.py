"""TimesFM agent wrapper.

TimesFM (Das et al., ICML 2024) is Google's decoder-only TS foundation model
trained on ~100B real-world time series points. Notably different architecture
from Chronos (encoder-decoder T5) so its forecasts should be substantively
decorrelated from Chronos-Bolt.

Reference:
  Das et al., "A Decoder-Only Foundation Model for Time-Series Forecasting",
  ICML 2024. https://arxiv.org/abs/2310.10688
  Repo: https://github.com/google-research/timesfm
  Weights: google/timesfm-2.0-500m-pytorch (200M params, 16k context)

TimesFM's experimental_quantile_forecast returns quantiles at fixed levels
[0.1, 0.2, ..., 0.9]. We interpolate linearly to whatever levels the caller
requests.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Agent, Forecast


# TimesFM's hard-coded quantile output grid (per timesfm source).
_TFM_QUANTILES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)


class TimesFMAgent(Agent):
    """Wraps `timesfm.TimesFm` with our Agent interface.

    The `frequency_input` knob (0 high freq, 1 medium, 2 low) signals TimesFM
    to use different positional encodings. Choose 0 for ETTh/ETTm (high-freq),
    1 for daily, 2 for monthly.
    """

    nominal_cost = 1.0

    def __init__(
        self,
        repo_id: str = "google/timesfm-2.0-500m-pytorch",
        backend: Optional[str] = None,
        per_core_batch_size: int = 64,
        horizon_len: int = 720,
        context_len: int = 512,
        frequency: int = 0,
        name_suffix: str = "",
    ):
        import torch
        import timesfm  # type: ignore

        if backend is None:
            backend = "gpu" if torch.cuda.is_available() else "cpu"

        # Different timesfm releases accept slightly different hparam kwargs.
        # Try the full set first; fall back to the minimal set if any kwarg
        # is rejected.
        hparam_kwargs = dict(
            backend=backend,
            per_core_batch_size=per_core_batch_size,
            horizon_len=horizon_len,
            context_len=context_len,
            num_layers=50,
            use_positional_embedding=False,
        )
        try:
            hparams = timesfm.TimesFmHparams(**hparam_kwargs)
        except TypeError:
            minimal = {k: hparam_kwargs[k] for k in ("backend", "per_core_batch_size", "horizon_len", "context_len")}
            hparams = timesfm.TimesFmHparams(**minimal)
        self._tfm = timesfm.TimesFm(
            hparams=hparams,
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=repo_id),
        )
        self._frequency = frequency
        self.name = f"timesfm{name_suffix}"
        self.repo_id = repo_id

    def forecast(
        self,
        history: np.ndarray,
        horizon: int,
        levels: np.ndarray,
        context: Optional[dict] = None,
    ) -> Forecast:
        history = np.asarray(history, dtype=np.float32)
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
        batch_size: int = 64,  # TimesFM has its own per_core_batch_size
    ) -> np.ndarray:
        histories = np.asarray(histories, dtype=np.float32)
        assert histories.ndim == 2, f"expected (N, L), got {histories.shape}"
        N = histories.shape[0]
        Q = len(levels)
        out = np.empty((N, Q, horizon), dtype=np.float32)
        # TimesFM `forecast` is itself batched; just pass everything.
        forecast_input = [histories[i] for i in range(N)]
        freq_input = [self._frequency] * N
        _point, quantile_forecast = self._tfm.forecast(forecast_input, freq=freq_input)
        # quantile_forecast: (N, max_horizon, 9) at fixed levels [0.1, ..., 0.9]
        q_native = np.asarray(quantile_forecast, dtype=np.float32)
        if q_native.shape[1] < horizon:
            raise RuntimeError(
                f"TimesFM produced horizon {q_native.shape[1]} < requested {horizon}; "
                "increase horizon_len at agent init."
            )
        q_native = q_native[:, :horizon, :]  # (N, horizon, 9)
        # Interpolate native levels [0.1..0.9] to requested levels.
        # For each (window, step), interp along the quantile axis.
        # Reshape to (N*H, 9), interp at requested levels, reshape back.
        flat = q_native.reshape(-1, q_native.shape[-1])  # (N*H, 9)
        interp = np.stack(
            [np.interp(levels, _TFM_QUANTILES, flat[i]) for i in range(flat.shape[0])],
            axis=0,
        )  # (N*H, Q)
        out_flat = interp.reshape(N, horizon, Q)
        out = np.transpose(out_flat, (0, 2, 1)).astype(np.float32)  # (N, Q, H)
        return out

    def _forecast_uni(self, x: np.ndarray, horizon: int, levels: np.ndarray) -> np.ndarray:
        return self.forecast_batch(x[None, :], horizon, levels)[0]
