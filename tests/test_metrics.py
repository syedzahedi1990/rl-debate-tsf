"""Unit tests for metrics — verify formulas with hand-computed cases."""

import numpy as np
import pytest

from distdeb.eval.metrics import mae, mse, pinball_loss, quantile_crps, empirical_coverage


def test_mse_mae_simple():
    p = np.array([1.0, 2.0, 3.0])
    t = np.array([1.0, 0.0, 6.0])
    assert mse(p, t) == pytest.approx((0 + 4 + 9) / 3)
    assert mae(p, t) == pytest.approx((0 + 2 + 3) / 3)


def test_pinball_at_median_equals_half_mae():
    rng = np.random.default_rng(0)
    p = rng.normal(size=100)
    t = rng.normal(size=100)
    # pinball(q=0.5, y, 0.5) = 0.5 * |y - q|
    assert np.mean(pinball_loss(p, t, 0.5)) == pytest.approx(0.5 * mae(p, t))


def test_crps_perfect_forecast_is_zero():
    target = np.array([0.0, 1.0, 2.0])
    levels = np.array([0.1, 0.5, 0.9])
    quantiles = np.stack([target, target, target])  # degenerate at truth
    assert quantile_crps(quantiles, levels, target) == pytest.approx(0.0)


def test_empirical_coverage_bounds():
    target = np.array([0.0, 1.0, 2.0])
    levels = np.array([0.1, 0.5, 0.9])
    quantiles = np.stack([target - 1, target, target + 1])  # 0.1 below, 0.9 above
    # All points fall inside [target-1, target+1]
    assert empirical_coverage(quantiles, levels, target, alpha=0.8) == pytest.approx(1.0)


def test_empirical_coverage_zero():
    target = np.array([0.0, 1.0, 2.0])
    levels = np.array([0.1, 0.5, 0.9])
    # Interval entirely above target.
    quantiles = np.stack([target + 10, target + 11, target + 12])
    assert empirical_coverage(quantiles, levels, target, alpha=0.8) == pytest.approx(0.0)
