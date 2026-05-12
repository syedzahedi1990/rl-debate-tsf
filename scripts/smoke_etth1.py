"""End-to-end smoke test on ETTh1 with the ARIMA agent.

Verifies:
  - Data loader produces correctly-shaped test windows.
  - ARIMA agent returns a Forecast with the requested quantile levels.
  - Eval (MSE/MAE/CRPS/coverage) runs and returns finite numbers.

This is the gate before adding LLM agents. If this fails on a laptop, the
Colab run will fail too.

Usage:
  python scripts/smoke_etth1.py --windows 32 --horizon 96
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Make `distdeb` importable when run from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distdeb.agents.arima_agent import ARIMAAgent
from distdeb.data.loaders import load_etth1_windows, iter_test_windows
from distdeb.eval.metrics import mae, mse, quantile_crps, empirical_coverage


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="dataset/ETT-small/ETTh1.csv")
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--windows", type=int, default=32, help="Test windows to evaluate.")
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"[err] {args.csv} not found. Run scripts/download_etth1.py first.")
        return 1

    series = load_etth1_windows(args.csv, seq_len=args.seq_len, pred_len=args.horizon, features="S")
    print(f"[data] z-scored series shape={series.data.shape}, train_end={series.train_end}, val_end={series.val_end}")

    agent = ARIMAAgent(order=(2, 1, 1))
    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

    medians, targets, all_quantiles = [], [], []
    t0 = time.perf_counter()
    for i, (history, target) in enumerate(
        iter_test_windows(series, seq_len=args.seq_len, pred_len=args.horizon, stride=args.stride, limit=args.windows)
    ):
        # Univariate smoke — squeeze trailing axis.
        hist_u = history.squeeze(-1)
        tgt_u = target.squeeze(-1)
        fc = agent.forecast(hist_u, horizon=args.horizon, levels=levels)
        medians.append(fc.median)
        all_quantiles.append(fc.quantiles)
        targets.append(tgt_u)
        if (i + 1) % 10 == 0:
            print(f"  window {i + 1}/{args.windows}  elapsed={time.perf_counter() - t0:.1f}s")

    median_arr = np.stack(medians)  # (N, H)
    target_arr = np.stack(targets)  # (N, H)
    quant_arr = np.stack(all_quantiles, axis=1)  # (Q, N, H)

    out = {
        "MSE": mse(median_arr, target_arr),
        "MAE": mae(median_arr, target_arr),
        "CRPS": quantile_crps(quant_arr, levels, target_arr),
        "coverage_80": empirical_coverage(quant_arr, levels, target_arr, alpha=0.8),
        "n_windows": int(median_arr.shape[0]),
        "elapsed_s": time.perf_counter() - t0,
    }
    print("\n[smoke results]")
    for k, v in out.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # Sanity checks — fail loudly if anything is wrong.
    assert np.isfinite(out["MSE"]) and np.isfinite(out["CRPS"])
    assert 0.0 <= out["coverage_80"] <= 1.0
    assert out["n_windows"] == args.windows
    print("\n[ok] smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
