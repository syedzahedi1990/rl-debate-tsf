"""Split-conformal calibration for quantile forecasts.

For a target coverage 1 - α, the standard split conformal procedure:
  1. On a held-out calibration set, compute the per-window non-conformity
     score s_i = max(q_{lo}(x_i) - y_i, y_i - q_{hi}(x_i)), aggregated
     across the horizon (we use the per-window mean).
  2. Take the empirical (1 - α) quantile c of these scores.
  3. On test, expand the interval: q_{lo}^calib = q_{lo} - c, q_{hi}^calib = q_{hi} + c.

Under exchangeability between calibration and test, this gives marginal
coverage >= 1 - α on test. We additionally calibrate intermediate quantiles
(0.25, 0.75) by a linearly-interpolated factor for CRPS evaluation.

Reference:
  Vovk et al., "Algorithmic Learning in a Random World", 2005.
  Romano et al., "Conformalized Quantile Regression", NeurIPS 2019.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def split_conformal_calibrate(
    val_quantiles: np.ndarray,   # (N_val, Q, H)
    val_targets: np.ndarray,     # (N_val, H)
    test_quantiles: np.ndarray,  # (N_test, Q, H)
    levels: np.ndarray,
    target_coverage: float = 0.8,
    aggregate: str = "mean",
) -> Tuple[np.ndarray, float]:
    """Conformalize the central (target_coverage)-interval.

    Returns (calibrated_test_quantiles, c).

    The middle quantile (0.5) is left untouched; the lo / hi quantiles of the
    target interval are shifted outward by `c`; intermediate quantiles are
    shifted by `c * (level - 0.5) / (hi_level - 0.5)` to preserve monotonicity
    of the quantile curve.
    """
    levels = np.asarray(levels, dtype=np.float32)
    lo_level = (1 - target_coverage) / 2  # e.g. 0.1 for 0.8 coverage
    hi_level = 1 - lo_level                # e.g. 0.9 for 0.8 coverage
    lo_idx = int(np.argmin(np.abs(levels - lo_level)))
    hi_idx = int(np.argmin(np.abs(levels - hi_level)))

    val_lo = val_quantiles[:, lo_idx, :]   # (N_val, H)
    val_hi = val_quantiles[:, hi_idx, :]
    # Per-(window, step) non-conformity score: max miscoverage.
    scores = np.maximum(val_lo - val_targets, val_targets - val_hi)
    # Per-window aggregation.
    if aggregate == "mean":
        scores_per_window = np.mean(scores, axis=1)
    elif aggregate == "max":
        scores_per_window = np.max(scores, axis=1)
    else:
        raise ValueError(f"unknown aggregate {aggregate!r}")
    # Empirical quantile of scores at target coverage.
    c = float(np.quantile(scores_per_window, target_coverage))

    calibrated = test_quantiles.copy()
    # Shift intermediate quantiles smoothly to keep monotonicity.
    for qi, lv in enumerate(levels):
        if lv <= 0.5:
            # Negative shift, scaled by distance from 0.5.
            scale = (0.5 - lv) / (0.5 - lo_level)
            calibrated[:, qi, :] -= c * scale
        else:
            scale = (lv - 0.5) / (hi_level - 0.5)
            calibrated[:, qi, :] += c * scale
    return calibrated, c
