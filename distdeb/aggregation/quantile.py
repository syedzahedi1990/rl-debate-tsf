"""Quantile aggregation operators.

The RL orchestrator's `combiner_update` action picks weights in the simplex
that feed `weighted_quantile_average`. For the Gate-1 baseline we use the
equal-weight variant — the dumbest possible "ensemble" that any RL policy
must beat.

Quantile averaging is a standard technique in distributional forecasting
(GluonTS `Combinator`, Chronos paper Appendix). It preserves monotonicity
of the quantile curve when inputs are monotone.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..agents.base import Forecast


def _check_alignment(forecasts: Sequence[Forecast]) -> np.ndarray:
    assert len(forecasts) > 0, "empty forecast list"
    levels = forecasts[0].levels
    shape = forecasts[0].quantiles.shape
    for f in forecasts[1:]:
        assert np.array_equal(f.levels, levels), "all forecasts must share quantile levels"
        assert f.quantiles.shape == shape, f"shape mismatch: {f.quantiles.shape} vs {shape}"
    return levels


def equal_weight_quantile_average(forecasts: Sequence[Forecast]) -> Forecast:
    """Per-quantile mean across agents."""
    levels = _check_alignment(forecasts)
    stacked = np.stack([f.quantiles for f in forecasts], axis=0)  # (N, Q, ...)
    avg = stacked.mean(axis=0)
    return Forecast(
        quantiles=avg,
        levels=levels,
        agent_name="equal_weight(" + ",".join(f.agent_name for f in forecasts) + ")",
        cost=sum(f.cost for f in forecasts),
    )


def weighted_quantile_average(forecasts: Sequence[Forecast], weights: Sequence[float]) -> Forecast:
    """Convex combination of quantile forecasts.

    weights are renormalized to the simplex. Negative weights raise.
    """
    levels = _check_alignment(forecasts)
    w = np.asarray(weights, dtype=np.float64)
    assert len(w) == len(forecasts), f"weights ({len(w)}) != forecasts ({len(forecasts)})"
    assert np.all(w >= 0), "weights must be non-negative"
    s = w.sum()
    assert s > 0, "weights must not sum to zero"
    w = w / s
    stacked = np.stack([f.quantiles for f in forecasts], axis=0)  # (N, Q, ...)
    avg = np.tensordot(w, stacked, axes=([0], [0]))
    return Forecast(
        quantiles=avg,
        levels=levels,
        agent_name="weighted(" + ",".join(f.agent_name for f in forecasts) + ")",
        cost=sum(f.cost for f in forecasts),
    )
