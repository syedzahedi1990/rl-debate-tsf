"""Gate 1 pilot: premise check for Distributional Debate.

Per DESIGN.md §8 — Gate 1 asks: does ANY multi-agent combination beat the
best single agent on CRPS at matched cost? If equal-weight ensemble of
{ARIMA, Chronos-Bolt} doesn't beat the better of the two on CRPS, the case
for an RL-orchestrated debate on top is weak.

This is a deliberately weak test on purpose: equal-weight ensembling is the
lowest-effort form of "debate-like aggregation". An RL orchestrator should
do strictly better; but if even the dumb ensemble fails, that's our signal
to pivot the framing.

Outputs:
  - results/pilot_gate1.json: full metric breakdown
  - stdout: human-readable summary + Gate 1 verdict

Usage on Colab:
  python scripts/pilot_gate1.py --windows -1 --horizon 96
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict

import numpy as np

# Make `distdeb` importable when run from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distdeb.agents.arima_agent import ARIMAAgent
from distdeb.agents.chronos_agent import ChronosBoltAgent
from distdeb.data.loaders import iter_test_windows, load_etth1_windows
from distdeb.eval.metrics import empirical_coverage, mae, mse, quantile_crps


def _metrics(quantiles: np.ndarray, levels: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """quantiles: (N, Q, H); targets: (N, H)."""
    median_idx = int(np.argmin(np.abs(levels - 0.5)))
    median = quantiles[:, median_idx, :]  # (N, H)
    # quantile_crps expects (Q, ...) with trailing axes matching target -> reshape to (Q, N, H)
    q_for_crps = np.transpose(quantiles, (1, 0, 2))
    return {
        "MSE": mse(median, targets),
        "MAE": mae(median, targets),
        "CRPS": quantile_crps(q_for_crps, levels, targets),
        "coverage_80": empirical_coverage(q_for_crps, levels, targets, alpha=0.8),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="dataset/ETT-small/ETTh1.csv")
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument(
        "--windows",
        type=int,
        default=-1,
        help="Number of test windows. -1 = full test split (~5800 for ETTh1 H=96).",
    )
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--chronos-model", default="amazon/chronos-bolt-base")
    p.add_argument("--chronos-batch", type=int, default=128)
    p.add_argument("--arima-order", default="2,1,1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/pilot_gate1.json")
    p.add_argument("--skip-arima", action="store_true", help="Skip ARIMA (it's slow on full set).")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"[err] {args.csv} not found. Run scripts/download_etth1.py first.")
        return 1

    np.random.seed(args.seed)
    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

    series = load_etth1_windows(args.csv, seq_len=args.seq_len, pred_len=args.horizon, features="S")
    print(f"[data] z-scored series shape={series.data.shape}, train_end={series.train_end}, val_end={series.val_end}")

    limit = None if args.windows == -1 else args.windows
    windows = list(iter_test_windows(series, args.seq_len, args.horizon, args.stride, limit=limit))
    print(f"[data] {len(windows)} test windows (stride={args.stride})")

    histories = np.stack([h.squeeze(-1) for h, _ in windows]).astype(np.float32)  # (N, L)
    targets = np.stack([t.squeeze(-1) for _, t in windows]).astype(np.float32)  # (N, H)

    agent_quantiles: Dict[str, np.ndarray] = {}
    timings: Dict[str, float] = {}

    if not args.skip_arima:
        order = tuple(int(x) for x in args.arima_order.split(","))
        print(f"[agent] ARIMA{order} on {len(histories)} windows...")
        arima = ARIMAAgent(order=order)
        t0 = time.perf_counter()
        arima_q = np.stack([arima.forecast(h, args.horizon, levels).quantiles for h in histories])
        timings["arima"] = time.perf_counter() - t0
        agent_quantiles["arima"] = arima_q.astype(np.float32)
        print(f"  done in {timings['arima']:.1f}s")

    print(f"[agent] Chronos-Bolt ({args.chronos_model}), batch={args.chronos_batch}...")
    chronos = ChronosBoltAgent(model_id=args.chronos_model)
    t0 = time.perf_counter()
    chronos_q = chronos.forecast_batch(histories, args.horizon, levels, batch_size=args.chronos_batch)
    timings["chronos"] = time.perf_counter() - t0
    agent_quantiles["chronos"] = chronos_q
    print(f"  done in {timings['chronos']:.1f}s")

    # Equal-weight ensemble of all available agents.
    if len(agent_quantiles) >= 2:
        ensemble_q = np.mean(np.stack(list(agent_quantiles.values()), axis=0), axis=0)
    else:
        ensemble_q = None

    results = {
        "config": {k: getattr(args, k) for k in vars(args)},
        "n_windows": len(windows),
        "agents": {name: _metrics(q, levels, targets) | {"wall_s": timings[name]} for name, q in agent_quantiles.items()},
    }
    if ensemble_q is not None:
        results["agents"]["equal_weight_ensemble"] = _metrics(ensemble_q, levels, targets)

    # Pretty print
    print("\n=== Gate 1 results ===")
    print(f"{'agent':25s} {'MSE':>8s} {'MAE':>8s} {'CRPS':>8s} {'cov80':>8s} {'wall_s':>8s}")
    for name, r in results["agents"].items():
        ws = r.get("wall_s", float("nan"))
        print(f"{name:25s} {r['MSE']:8.4f} {r['MAE']:8.4f} {r['CRPS']:8.4f} {r['coverage_80']:8.4f} {ws:8.1f}")

    # Verdict
    if ensemble_q is not None:
        single_crps = {n: r["CRPS"] for n, r in results["agents"].items() if n != "equal_weight_ensemble"}
        best_single_name = min(single_crps, key=single_crps.get)
        best_single = single_crps[best_single_name]
        ens = results["agents"]["equal_weight_ensemble"]["CRPS"]
        delta = 100 * (best_single - ens) / best_single
        results["gate1"] = {
            "best_single_agent": best_single_name,
            "best_single_CRPS": best_single,
            "ensemble_CRPS": ens,
            "rel_improvement_pct": delta,
            "pass": ens < best_single,
        }
        print(f"\nbest single: {best_single_name} (CRPS={best_single:.4f})")
        print(f"ensemble  :              CRPS={ens:.4f}")
        if ens < best_single:
            print(f"GATE 1 PASS: equal-weight ensemble beats best single by {delta:+.2f}% CRPS")
        else:
            print(f"GATE 1 FAIL: ensemble is {-delta:.2f}% worse than best single on CRPS")
            print("  -> premise weak; revisit panel composition before building RL orchestrator")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[saved] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
