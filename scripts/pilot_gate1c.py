"""Gate 1c: validation-tuned weighted ensemble.

Gate 1's original verdict (equal-weight FAIL on 3/4 ETT datasets) was a
predictable consequence of mismatched-skill agents under uniform weighting.
The fair premise check is: does the validation-optimal linear combination
beat the best single agent?

This script:
  1. Builds a panel: ARIMA + Chronos-Bolt-base + Chronos-Bolt-tiny.
  2. For each ETT dataset, computes each agent's quantile forecasts on the
     validation split (256 windows) and the test split (1024 windows by
     default; -1 for all).
  3. Grid-searches the simplex of weights on val CRPS (resolution 0.1 in
     each dimension — 66 combinations for 3 agents).
  4. Applies the val-best weights on test and reports test CRPS, MAE,
     coverage_80, and chosen weights.

Outcomes interpreted in DESIGN.md §8 Gate 1:
  - Optimal weights non-trivial (multiple > 0.1) AND test gain > best
    single -> green light for RL.
  - Optimal weights collapse to one agent -> panel too narrow,
    diversify before RL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import product
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distdeb.agents.arima_agent import ARIMAAgent
from distdeb.agents.chronos_agent import ChronosBoltAgent
from distdeb.eval.metrics import empirical_coverage, mae, mse, quantile_crps


DATASETS = [
    ("ETTh1", "dataset/ETT-small/ETTh1.csv", "hour"),
    ("ETTh2", "dataset/ETT-small/ETTh2.csv", "hour"),
    ("ETTm1", "dataset/ETT-small/ETTm1.csv", "minute"),
    ("ETTm2", "dataset/ETT-small/ETTm2.csv", "minute"),
]


def _ett_boundaries(freq):
    if freq == "hour":
        return 12 * 30 * 24, 16 * 30 * 24
    if freq == "minute":
        return 12 * 30 * 24 * 4, 16 * 30 * 24 * 4
    raise ValueError(freq)


def _load_ett(csv_path, freq, target="OT"):
    import pandas as pd
    df = pd.read_csv(csv_path).drop(columns=["date"], errors="ignore")
    arr = df[[target]].values.astype(np.float32)
    train_end, val_end = _ett_boundaries(freq)
    mean = arr[:train_end].mean(axis=0)
    std = arr[:train_end].std(axis=0) + 1e-8
    return {"data": (arr - mean) / std, "train_end": train_end, "val_end": val_end}


def _windows(series, start_idx, end_idx, seq_len, pred_len, stride, limit):
    data = series["data"]
    n = 0
    for i in range(start_idx, end_idx, stride):
        if i - seq_len < 0:
            continue
        yield data[i - seq_len : i], data[i : i + pred_len]
        n += 1
        if limit is not None and n >= limit:
            break


def val_windows(series, seq_len, pred_len, stride, limit):
    start = series["train_end"] + seq_len
    end = series["val_end"] - pred_len + 1
    return list(_windows(series, start, end, seq_len, pred_len, stride, limit))


def test_windows(series, seq_len, pred_len, stride, limit):
    start = series["val_end"]
    end = len(series["data"]) - pred_len + 1
    return list(_windows(series, start, end, seq_len, pred_len, stride, limit))


def _crps_of_mixture(weights: np.ndarray, agent_q_stack: np.ndarray, levels: np.ndarray, targets: np.ndarray) -> float:
    """weights: (A,); agent_q_stack: (A, N, Q, H); targets: (N, H)."""
    combined = np.einsum("a,anqh->nqh", weights, agent_q_stack)
    return quantile_crps(np.transpose(combined, (1, 0, 2)), levels, targets)


def _metrics_for(q: np.ndarray, levels: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    median_idx = int(np.argmin(np.abs(levels - 0.5)))
    median = q[:, median_idx, :]
    q_for_crps = np.transpose(q, (1, 0, 2))
    return {
        "MSE": mse(median, targets),
        "MAE": mae(median, targets),
        "CRPS": quantile_crps(q_for_crps, levels, targets),
        "coverage_80": empirical_coverage(q_for_crps, levels, targets, alpha=0.8),
    }


def grid_search_simplex(agent_q_stack: np.ndarray, levels: np.ndarray, targets: np.ndarray, resolution: int = 10) -> Tuple[np.ndarray, float]:
    """Brute-force the simplex at resolution 1/resolution.

    For A=3 and resolution=10, evaluates 66 combinations (the C(A+r-1, A-1)
    triangular number). For A=2, 11 combinations.
    """
    A = agent_q_stack.shape[0]
    best_w, best_crps = None, float("inf")
    if A == 2:
        grid = [(w, 1 - w) for w in np.linspace(0, 1, resolution + 1)]
    elif A == 3:
        grid = []
        for i in range(resolution + 1):
            for j in range(resolution + 1 - i):
                k = resolution - i - j
                grid.append((i / resolution, j / resolution, k / resolution))
    else:
        # Random Dirichlet samples if more agents — cheap enough.
        rng = np.random.default_rng(0)
        grid = list(rng.dirichlet(np.ones(A), size=200))
    for w in grid:
        w = np.asarray(w, dtype=np.float64)
        c = _crps_of_mixture(w, agent_q_stack, levels, targets)
        if c < best_crps:
            best_crps, best_w = c, w
    return best_w, best_crps


def run_dataset(name, csv_path, freq, args, agents: Dict[str, object], levels: np.ndarray):
    if not os.path.exists(csv_path):
        print(f"[skip] {csv_path}")
        return None
    print(f"\n--- {name} ---")
    series = _load_ett(csv_path, freq)
    print(f"[data] shape={series['data'].shape} train_end={series['train_end']} val_end={series['val_end']}")

    val_w = val_windows(series, args.seq_len, args.horizon, args.stride, args.val_windows)
    test_w = test_windows(series, args.seq_len, args.horizon, args.stride, None if args.windows == -1 else args.windows)
    print(f"[data] val={len(val_w)} test={len(test_w)}")

    val_hist = np.stack([h.squeeze(-1) for h, _ in val_w]).astype(np.float32)
    val_tgt = np.stack([t.squeeze(-1) for _, t in val_w]).astype(np.float32)
    test_hist = np.stack([h.squeeze(-1) for h, _ in test_w]).astype(np.float32)
    test_tgt = np.stack([t.squeeze(-1) for _, t in test_w]).astype(np.float32)

    # Get forecasts for each agent on val and test.
    val_q: Dict[str, np.ndarray] = {}
    test_q: Dict[str, np.ndarray] = {}
    for an, a in agents.items():
        print(f"[agent] {an}...")
        t0 = time.perf_counter()
        if hasattr(a, "forecast_batch"):
            val_q[an] = a.forecast_batch(val_hist, args.horizon, levels, batch_size=args.chronos_batch)
            test_q[an] = a.forecast_batch(test_hist, args.horizon, levels, batch_size=args.chronos_batch)
        else:
            val_q[an] = np.stack([a.forecast(h, args.horizon, levels).quantiles for h in val_hist]).astype(np.float32)
            test_q[an] = np.stack([a.forecast(h, args.horizon, levels).quantiles for h in test_hist]).astype(np.float32)
        print(f"  {time.perf_counter() - t0:.1f}s")

    agent_names = list(agents.keys())
    val_stack = np.stack([val_q[n] for n in agent_names], axis=0)  # (A, N_val, Q, H)
    test_stack = np.stack([test_q[n] for n in agent_names], axis=0)

    # Per-agent test metrics
    per_agent = {n: _metrics_for(test_q[n], levels, test_tgt) for n in agent_names}

    # Uniform ensemble test metrics
    uniform = np.mean(test_stack, axis=0)
    uniform_m = _metrics_for(uniform, levels, test_tgt)

    # Val-tuned weights, then test metrics
    best_w, val_best_crps = grid_search_simplex(val_stack, levels, val_tgt, resolution=args.grid_resolution)
    test_combined = np.einsum("a,anqh->nqh", best_w, test_stack)
    tuned_m = _metrics_for(test_combined, levels, test_tgt)

    row = {
        "dataset": name,
        "agents": agent_names,
        "val_best_weights": [float(x) for x in best_w],
        "val_best_crps": float(val_best_crps),
        "per_agent_test": per_agent,
        "uniform_test": uniform_m,
        "val_tuned_test": tuned_m,
    }

    print(f"\n  val-best weights: {dict(zip(agent_names, [round(float(x), 2) for x in best_w]))}")
    print(f"  {'method':25s} {'MSE':>8s} {'MAE':>8s} {'CRPS':>8s} {'cov80':>8s}")
    for n in agent_names:
        m = per_agent[n]
        print(f"  {n:25s} {m['MSE']:8.4f} {m['MAE']:8.4f} {m['CRPS']:8.4f} {m['coverage_80']:8.4f}")
    print(f"  {'uniform':25s} {uniform_m['MSE']:8.4f} {uniform_m['MAE']:8.4f} {uniform_m['CRPS']:8.4f} {uniform_m['coverage_80']:8.4f}")
    print(f"  {'val-tuned':25s} {tuned_m['MSE']:8.4f} {tuned_m['MAE']:8.4f} {tuned_m['CRPS']:8.4f} {tuned_m['coverage_80']:8.4f}")

    best_single_name = min(agent_names, key=lambda n: per_agent[n]["CRPS"])
    best_single_crps = per_agent[best_single_name]["CRPS"]
    delta = 100 * (best_single_crps - tuned_m["CRPS"]) / best_single_crps
    row["best_single"] = best_single_name
    row["delta_vs_best_single_pct"] = float(delta)
    row["pass"] = bool(tuned_m["CRPS"] < best_single_crps)
    print(f"  -> val-tuned vs best({best_single_name}): {delta:+.2f}%  {'PASS' if row['pass'] else 'FAIL'}")

    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--windows", type=int, default=1024, help="Test windows per dataset; -1 for all.")
    p.add_argument("--val-windows", type=int, default=256)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--chronos-base", default="amazon/chronos-bolt-base")
    p.add_argument("--chronos-tiny", default="amazon/chronos-bolt-tiny")
    p.add_argument("--chronos-batch", type=int, default=128)
    p.add_argument("--arima-order", default="2,1,1")
    p.add_argument("--skip-arima", action="store_true")
    p.add_argument("--grid-resolution", type=int, default=10)
    p.add_argument("--out", default="results/pilot_gate1c.json")
    args = p.parse_args()

    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    print(f"[setup] loading Chronos-Bolt base + tiny once")
    agents: Dict[str, object] = {}
    if not args.skip_arima:
        agents["arima"] = ARIMAAgent(order=tuple(int(x) for x in args.arima_order.split(",")))
    agents["chronos_base"] = ChronosBoltAgent(model_id=args.chronos_base, name_suffix="_base")
    agents["chronos_tiny"] = ChronosBoltAgent(model_id=args.chronos_tiny, name_suffix="_tiny")

    rows = [run_dataset(n, p_, f, args, agents, levels) for (n, p_, f) in DATASETS]
    rows = [r for r in rows if r]

    # Aggregate verdict
    n_pass = sum(1 for r in rows if r["pass"])
    print("\n\n=== Gate 1c aggregate ===")
    print(f"val-tuned beats best-single on {n_pass}/{len(rows)} datasets")
    print(f"{'dataset':10s} {'best_single':14s} {'delta%':>8s} {'weights (' + ','.join(list(agents.keys())) + ')':50s}")
    for r in rows:
        w = ",".join(f"{x:.2f}" for x in r["val_best_weights"])
        print(f"{r['dataset']:10s} {r['best_single']:14s} {r['delta_vs_best_single_pct']:+8.2f} ({w})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2, default=float)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    sys.exit(main())
