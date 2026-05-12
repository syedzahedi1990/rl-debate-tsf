"""Smoke-test a single agent in isolation.

Useful when adding a new agent (TimesFM, Moirai, LLMTime, ...) before wiring
it into the full Gate-1 panel run. Confirms the agent:
  - constructs without error
  - returns finite quantile forecasts of the expected shape
  - cache plumbing works (second call should be near-instant)

Usage:
  python scripts/smoke_agent.py --agent timesfm
  python scripts/smoke_agent.py --agent chronos_base
  python scripts/smoke_agent.py --agent timesfm --windows 64 --horizon 96
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distdeb.data.loaders import iter_test_windows, load_etth1_windows
from distdeb.eval.metrics import empirical_coverage, mae, mse, quantile_crps
from distdeb.utils.cache import ForecastCache, cached_forecast


def build_agent(name: str):
    if name == "chronos_base":
        from distdeb.agents.chronos_agent import ChronosBoltAgent
        return ChronosBoltAgent(model_id="amazon/chronos-bolt-base")
    if name == "chronos_tiny":
        from distdeb.agents.chronos_agent import ChronosBoltAgent
        return ChronosBoltAgent(model_id="amazon/chronos-bolt-tiny")
    if name == "timesfm":
        from distdeb.agents.timesfm_agent import TimesFMAgent
        return TimesFMAgent(repo_id="google/timesfm-2.0-500m-pytorch")
    raise ValueError(f"unknown agent: {name!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True, help="chronos_base | chronos_tiny | timesfm | ...")
    p.add_argument("--csv", default="dataset/ETT-small/ETTh1.csv")
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--windows", type=int, default=64)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--cache-root", default="data_cache/forecasts")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"[err] {args.csv} not found. Run scripts/download_etth1.py first.")
        return 1

    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    series = load_etth1_windows(args.csv, seq_len=args.seq_len, pred_len=args.horizon, features="S")
    windows = list(iter_test_windows(series, args.seq_len, args.horizon, args.stride, limit=args.windows))
    histories = np.stack([h.squeeze(-1) for h, _ in windows]).astype(np.float32)
    targets = np.stack([t.squeeze(-1) for _, t in windows]).astype(np.float32)

    print(f"[setup] {args.agent}, {len(histories)} windows, H={args.horizon}")
    t0 = time.perf_counter()
    agent = build_agent(args.agent)
    print(f"[setup] agent constructed in {time.perf_counter() - t0:.1f}s; name={agent.name}")

    cache = ForecastCache(root=args.cache_root)

    def compute(h, H, lv):
        return agent.forecast_batch(h, H, lv, batch_size=64)

    # First call — may or may not hit cache.
    t0 = time.perf_counter()
    q = cached_forecast(
        cache=cache,
        dataset="ETTh1",
        agent_name=agent.name,
        seq_len=args.seq_len,
        horizon=args.horizon,
        split="smoke",
        histories=histories,
        levels=levels,
        compute_fn=compute,
    )
    t_first = time.perf_counter() - t0

    # Second call — must hit cache.
    t0 = time.perf_counter()
    q2 = cached_forecast(
        cache=cache,
        dataset="ETTh1",
        agent_name=agent.name,
        seq_len=args.seq_len,
        horizon=args.horizon,
        split="smoke",
        histories=histories,
        levels=levels,
        compute_fn=compute,
    )
    t_second = time.perf_counter() - t0

    assert q.shape == (args.windows, len(levels), args.horizon), f"unexpected shape {q.shape}"
    assert np.allclose(q, q2)
    assert np.all(np.isfinite(q))

    median_idx = int(np.argmin(np.abs(levels - 0.5)))
    metrics = {
        "MSE": mse(q[:, median_idx, :], targets),
        "MAE": mae(q[:, median_idx, :], targets),
        "CRPS": quantile_crps(np.transpose(q, (1, 0, 2)), levels, targets),
        "coverage_80": empirical_coverage(np.transpose(q, (1, 0, 2)), levels, targets, alpha=0.8),
    }

    print(f"\n[smoke]    first call: {t_first:6.2f}s")
    print(f"[smoke]   cached call: {t_second:6.2f}s (must be near-instant)")
    print(f"[metrics]")
    for k, v in metrics.items():
        print(f"  {k:>14s}: {v:.6f}")
    print("\n[ok] smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
