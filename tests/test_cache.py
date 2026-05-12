"""Tests for forecast cache."""

import numpy as np
import pytest

from distdeb.utils.cache import ForecastCache, cached_forecast


@pytest.fixture
def cache(tmp_path):
    return ForecastCache(root=tmp_path / "fc")


def test_miss_then_hit(cache):
    levels = np.array([0.1, 0.5, 0.9])
    q = np.random.randn(10, 3, 96).astype(np.float32)
    args = dict(dataset="X", agent="A", seq_len=96, horizon=96, split="test")
    assert cache.load(**args, n_windows=10, levels=levels) is None
    cache.save(**args, quantiles=q, levels=levels)
    got = cache.load(**args, n_windows=10, levels=levels)
    assert got is not None
    assert np.allclose(got, q)


def test_prefix_load(cache):
    levels = np.array([0.1, 0.5, 0.9])
    q = np.random.randn(20, 3, 96).astype(np.float32)
    args = dict(dataset="X", agent="A", seq_len=96, horizon=96, split="test")
    cache.save(**args, quantiles=q, levels=levels)
    got = cache.load(**args, n_windows=5, levels=levels)
    assert got is not None
    assert got.shape == (5, 3, 96)
    assert np.allclose(got, q[:5])


def test_not_enough_windows_returns_none(cache):
    levels = np.array([0.1, 0.5, 0.9])
    q = np.random.randn(5, 3, 96).astype(np.float32)
    args = dict(dataset="X", agent="A", seq_len=96, horizon=96, split="test")
    cache.save(**args, quantiles=q, levels=levels)
    assert cache.load(**args, n_windows=10, levels=levels) is None


def test_level_mismatch_returns_none(cache):
    q = np.random.randn(10, 3, 96).astype(np.float32)
    args = dict(dataset="X", agent="A", seq_len=96, horizon=96, split="test")
    cache.save(**args, quantiles=q, levels=np.array([0.1, 0.5, 0.9]))
    other = cache.load(**args, n_windows=10, levels=np.array([0.25, 0.5, 0.75]))
    assert other is None


def test_cached_forecast_only_computes_once(cache):
    levels = np.array([0.1, 0.5, 0.9])
    histories = np.random.randn(8, 96).astype(np.float32)
    call_count = {"n": 0}

    def compute(h, H, lv):
        call_count["n"] += 1
        return np.random.randn(h.shape[0], len(lv), H).astype(np.float32)

    a = cached_forecast(
        cache=cache, dataset="D", agent_name="Z", seq_len=96, horizon=96, split="test",
        histories=histories, levels=levels, compute_fn=compute,
    )
    b = cached_forecast(
        cache=cache, dataset="D", agent_name="Z", seq_len=96, horizon=96, split="test",
        histories=histories, levels=levels, compute_fn=compute,
    )
    assert call_count["n"] == 1
    assert np.allclose(a, b)
