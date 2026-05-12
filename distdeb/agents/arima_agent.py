from __future__ import annotations

from typing import Optional

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from .base import Agent, Forecast


class ARIMAAgent(Agent):
    """Cheap statistical anchor. Fits ARIMA(p,d,q) per call.

    Used both as a baseline panel member and as the ablation that tests
    "Are LMs Actually Useful for TSF?" — if the orchestrator routes to this
    every time, we report that honestly.
    """

    name = "arima"
    nominal_cost = 0.01  # ~free relative to an LLM call

    def __init__(self, order=(2, 1, 1)):
        self.order = order

    def forecast(
        self,
        history: np.ndarray,
        horizon: int,
        levels: np.ndarray,
        context: Optional[dict] = None,
    ) -> Forecast:
        history = np.asarray(history).astype(float)
        if history.ndim == 2:
            # Univariate fit per channel; orchestrator handles multivariate composition.
            qs = np.stack(
                [self._forecast_univariate(history[:, v], horizon, levels) for v in range(history.shape[1])],
                axis=-1,
            )  # (Q, H, V)
        else:
            qs = self._forecast_univariate(history, horizon, levels)  # (Q, H)

        return Forecast(quantiles=qs, levels=np.asarray(levels), agent_name=self.name, cost=self.nominal_cost)

    def _forecast_univariate(self, x: np.ndarray, horizon: int, levels: np.ndarray) -> np.ndarray:
        try:
            model = ARIMA(x, order=self.order).fit(method_kwargs={"warn_convergence": False})
            fc = model.get_forecast(steps=horizon)
            mean = fc.predicted_mean  # (H,)
            se = np.sqrt(np.maximum(fc.var_pred_mean, 1e-12))  # (H,)
        except Exception:
            # Fall back to naive last-value with empirical residual std.
            mean = np.full(horizon, x[-1])
            se = np.full(horizon, np.std(np.diff(x)) if len(x) > 1 else 1.0)

        # Inverse normal quantiles for parametric quantile output.
        from scipy.stats import norm

        z = norm.ppf(levels)  # (Q,)
        qs = mean[None, :] + z[:, None] * se[None, :]  # (Q, H)
        return qs
