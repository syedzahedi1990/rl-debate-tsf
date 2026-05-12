"""Forecast metrics: point (MSE, MAE) + probabilistic (CRPS via quantile pinball, coverage).

Conventions:
  - `pred`: median forecast, shape (..., H) or (..., H, V)
  - `target`: ground truth, same shape as `pred`
  - `quantiles`: shape (Q, ...,) with leading axis = quantile levels
  - `levels`: 1-d array of quantile levels in (0, 1), strictly ascending
"""

from __future__ import annotations

import numpy as np


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def pinball_loss(q_pred: np.ndarray, target: np.ndarray, level: float) -> np.ndarray:
    """Per-element pinball loss at one quantile level.

    L(q, y, a) = (y - q) * a               if y >= q
                 (q - y) * (1 - a)         otherwise
    Returns same shape as q_pred / target.
    """
    diff = target - q_pred
    return np.maximum(level * diff, (level - 1.0) * diff)


def quantile_crps(quantiles: np.ndarray, levels: np.ndarray, target: np.ndarray) -> float:
    """CRPS approximation from a quantile forecast.

    Uses the standard quantile-CRPS estimator: average pinball loss over the
    provided levels, scaled by 2 / Q so that with dense quantile grids it
    converges to CRPS. Following Chronos and GluonTS conventions.

    quantiles: shape (Q, ...) — must align with `target` on trailing axes.
    levels: shape (Q,)
    target: shape (...,) matching `quantiles` trailing axes.
    """
    quantiles = np.asarray(quantiles)
    levels = np.asarray(levels)
    target = np.asarray(target)
    assert quantiles.shape[1:] == target.shape, (
        f"quantiles trailing shape {quantiles.shape[1:]} must match target {target.shape}"
    )
    losses = []
    for q in range(quantiles.shape[0]):
        losses.append(np.mean(pinball_loss(quantiles[q], target, float(levels[q]))))
    # 2/Q normalization makes the estimator a Riemann-sum approximation to CRPS.
    return 2.0 * float(np.mean(losses))


def empirical_coverage(
    quantiles: np.ndarray, levels: np.ndarray, target: np.ndarray, alpha: float = 0.8
) -> float:
    """Empirical coverage of the central alpha-interval.

    Picks the closest available quantile levels to (1-alpha)/2 and 1-(1-alpha)/2.
    Returns fraction of `target` elements falling inside.
    """
    lo_level = (1 - alpha) / 2
    hi_level = 1 - lo_level
    lo_idx = int(np.argmin(np.abs(levels - lo_level)))
    hi_idx = int(np.argmin(np.abs(levels - hi_level)))
    lo, hi = quantiles[lo_idx], quantiles[hi_idx]
    inside = (target >= lo) & (target <= hi)
    return float(np.mean(inside))
