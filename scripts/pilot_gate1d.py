"""Gate 1d: diversified Chronos panel + blocked-val weighting.

Diagnosis from Gate 1c:
  - Single-block 256-window val selection overfit (ETTh1 weights collapsed
    onto ARIMA on val despite ARIMA being 26% worse than Chronos on test).
  - ARIMA's huge skill gap from Chronos drags uniform ensembles wherever
    the gap is widest (ETTh2, ETTm2).
  - We need: similarly-skilled but decorrelated agents AND a more robust
    val protocol that doesn't trust a single regime block.

This script:
  1. Builds a Chronos-only panel: {chronos_base, chronos_base_detrend,
     chronos_base_diff, chronos_tiny}. Same backbone family -> similar
     headline skill. Different preprocessing -> decorrelated errors.
  2. Uses 4-block stratified val: 4 disjoint 64-window blocks spread
     across the val split. Weight selection minimizes the mean CRPS
     across blocks (variance reduction).
  3. Runs the simplex grid as before (resolution 10 -> 286 points for
     4 agents).
  4. Reports per-agent test metrics + uniform + val-tuned + best-single
     verdict per dataset.

Usage:
  python scripts/pilot_gate1d.py --windows 1024 --val-windows 64 --n-val-blocks 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def _make_windows(series, start_idx, end_idx, seq_len, pred_len, stride=1, limit=None):
    data = series["data"]
    n = 0
    out = []
    for i in range(start_idx, end_idx, stride):
        if i - seq_len < 0:
            continue
        out.append((data[i - seq_len : i], data[i : i + pred_len]))
        n += 1
        if limit is not None and n >= limit:
            break
    return out


def stratified_val_blocks(series, seq_len, pred_len, n_blocks: int, block_size: int):
    """n_blocks disjoint windows of `block_size` each, evenly spaced across val."""
    val_lo = series["train_end"] + seq_len
    val_hi = series["val_end"] - pred_len + 1
    block_span = max(block_size + 1, (val_hi - val_lo) // n_blocks)
    blocks: List[List] = []
    for b in range(n_blocks):
        start = val_lo + b * block_span
        end = min(start + block_size, val_hi)
        blocks.append(_make_windows(series, start, end, seq_len, pred_len))
    return blocks


def test_windows(series, seq_len, pred_len, stride, limit):
    start = series["val_end"]
    end = len(series["data"]) - pred_len + 1
    return _make_windows(series, start, end, seq_len, pred_len, stride, limit)


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


def _crps_of_mixture(weights, agent_q_stack, levels, targets):
    combined = np.einsum("a,anqh->nqh", weights, agent_q_stack)
    return quantile_crps(np.transpose(combined, (1, 0, 2)), levels, targets)


def simplex_grid(A: int, resolution: int):
    """All non-negative integer A-tuples summing to `resolution`. Returns a
    list of (resolution,)-summed integer tuples; user divides by resolution.
    """
    def recur(remaining: int, depth: int, partial: List[int], out: List):
        if depth == 1:
            out.append(tuple(partial + [remaining]))
            return
        for i in range(remaining + 1):
            recur(remaining - i, depth - 1, partial + [i], out)

    out: List[tuple] = []
    recur(resolution, A, [], out)
    return out


def grid_search_simplex_multiblock(agent_q_val_blocks, levels, val_tgt_blocks, resolution: int = 10):
    """agent_q_val_blocks: list of arrays (A, N_b, Q, H); val_tgt_blocks: list of (N_b, H).

    Searches the resolution-N simplex. Score: mean of per-block CRPS.
    Returns (best_weights, best_mean_crps, per_block_best).
    """
    A = agent_q_val_blocks[0].shape[0]
    grid = simplex_grid(A, resolution)
    best_w, best_score = None, float("inf")
    for tup in grid:
        w = np.array(tup, dtype=np.float64) / resolution
        per_block = [_crps_of_mixture(w, q, levels, t) for q, t in zip(agent_q_val_blocks, val_tgt_blocks)]
        score = float(np.mean(per_block))
        if score < best_score:
            best_score, best_w = score, w
    return best_w, best_score


def run_dataset(name, csv_path, freq, args, agents: Dict[str, object], levels: np.ndarray):
    if not os.path.exists(csv_path):
        return None
    print(f"\n--- {name} ---")
    series = _load_ett(csv_path, freq)
    val_blocks = stratified_val_blocks(series, args.seq_len, args.horizon, args.n_val_blocks, args.val_windows)
    test_w = test_windows(series, args.seq_len, args.horizon, args.stride, None if args.windows == -1 else args.windows)
    print(f"[data] val blocks: {[len(b) for b in val_blocks]}, test: {len(test_w)}")

    val_hist_blocks = [np.stack([h.squeeze(-1) for h, _ in b]).astype(np.float32) for b in val_blocks]
    val_tgt_blocks = [np.stack([t.squeeze(-1) for _, t in b]).astype(np.float32) for b in val_blocks]
    test_hist = np.stack([h.squeeze(-1) for h, _ in test_w]).astype(np.float32)
    test_tgt = np.stack([t.squeeze(-1) for _, t in test_w]).astype(np.float32)

    # Forecasts per agent on each val block and on test
    val_q_blocks: List[np.ndarray] = [None] * args.n_val_blocks  # type: ignore
    test_q: Dict[str, np.ndarray] = {}
    agent_names = list(agents.keys())
    for an in agent_names:
        a = agents[an]
        print(f"[agent] {an}...")
        t0 = time.perf_counter()
        test_q[an] = a.forecast_batch(test_hist, args.horizon, levels, batch_size=args.chronos_batch)
        for bi, vh in enumerate(val_hist_blocks):
            vq = a.forecast_batch(vh, args.horizon, levels, batch_size=args.chronos_batch)
            if val_q_blocks[bi] is None:
                val_q_blocks[bi] = vq[None]  # (1, N_b, Q, H)
            else:
                val_q_blocks[bi] = np.concatenate([val_q_blocks[bi], vq[None]], axis=0)
        print(f"  {time.perf_counter() - t0:.1f}s")

    test_stack = np.stack([test_q[n] for n in agent_names], axis=0)  # (A, N_test, Q, H)

    per_agent = {n: _metrics_for(test_q[n], levels, test_tgt) for n in agent_names}
    uniform = np.mean(test_stack, axis=0)
    uniform_m = _metrics_for(uniform, levels, test_tgt)

    best_w, val_mean_crps = grid_search_simplex_multiblock(val_q_blocks, levels, val_tgt_blocks, resolution=args.grid_resolution)
    tuned_test = np.einsum("a,anqh->nqh", best_w, test_stack)
    tuned_m = _metrics_for(tuned_test, levels, test_tgt)

    print(f"\n  val-best (4-block mean CRPS={val_mean_crps:.4f}) weights:")
    for n, w in zip(agent_names, best_w):
        print(f"    {n:25s} {float(w):.2f}")
    print(f"  {'method':25s} {'MSE':>8s} {'MAE':>8s} {'CRPS':>8s} {'cov80':>8s}")
    for n in agent_names:
        m = per_agent[n]
        print(f"  {n:25s} {m['MSE']:8.4f} {m['MAE']:8.4f} {m['CRPS']:8.4f} {m['coverage_80']:8.4f}")
    print(f"  {'uniform':25s} {uniform_m['MSE']:8.4f} {uniform_m['MAE']:8.4f} {uniform_m['CRPS']:8.4f} {uniform_m['coverage_80']:8.4f}")
    print(f"  {'val-tuned':25s} {tuned_m['MSE']:8.4f} {tuned_m['MAE']:8.4f} {tuned_m['CRPS']:8.4f} {tuned_m['coverage_80']:8.4f}")

    best_single_name = min(agent_names, key=lambda n: per_agent[n]["CRPS"])
    best_single_crps = per_agent[best_single_name]["CRPS"]
    d_uniform = 100 * (best_single_crps - uniform_m["CRPS"]) / best_single_crps
    d_tuned = 100 * (best_single_crps - tuned_m["CRPS"]) / best_single_crps
    print(f"  -> uniform vs best({best_single_name}): {d_uniform:+.2f}%  {'PASS' if d_uniform > 0 else 'FAIL'}")
    print(f"  -> val-tuned vs best({best_single_name}): {d_tuned:+.2f}%  {'PASS' if d_tuned > 0 else 'FAIL'}")

    return {
        "dataset": name,
        "agents": agent_names,
        "val_best_weights": [float(x) for x in best_w],
        "per_agent_test": per_agent,
        "uniform_test": uniform_m,
        "val_tuned_test": tuned_m,
        "best_single": best_single_name,
        "delta_uniform_pct": float(d_uniform),
        "delta_tuned_pct": float(d_tuned),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--windows", type=int, default=1024)
    p.add_argument("--val-windows", type=int, default=64, help="windows PER val block")
    p.add_argument("--n-val-blocks", type=int, default=4)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--chronos-base", default="amazon/chronos-bolt-base")
    p.add_argument("--chronos-tiny", default="amazon/chronos-bolt-tiny")
    p.add_argument("--chronos-batch", type=int, default=128)
    p.add_argument("--grid-resolution", type=int, default=10)
    p.add_argument("--out", default="results/pilot_gate1d.json")
    args = p.parse_args()

    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

    print("[setup] building Chronos-only panel (4 agents):")
    print("  chronos_base               (identity)")
    print("  chronos_base_detrend       (linear detrend specialist)")
    print("  chronos_base_diff          (first-difference specialist)")
    print("  chronos_tiny               (smaller backbone, identity)")

    agents = {
        "chronos_base": ChronosBoltAgent(model_id=args.chronos_base, preprocessor="identity"),
        "chronos_base_detrend": ChronosBoltAgent(model_id=args.chronos_base, preprocessor="detrend"),
        "chronos_base_diff": ChronosBoltAgent(model_id=args.chronos_base, preprocessor="diff"),
        "chronos_tiny": ChronosBoltAgent(model_id=args.chronos_tiny, preprocessor="identity"),
    }

    rows = []
    for n, p_, f in DATASETS:
        r = run_dataset(n, p_, f, args, agents, levels)
        if r:
            rows.append(r)

    print("\n\n=== Gate 1d aggregate ===")
    print(f"{'dataset':10s} {'best_single':25s} {'uniform%':>10s} {'tuned%':>10s}")
    for r in rows:
        print(f"{r['dataset']:10s} {r['best_single']:25s} {r['delta_uniform_pct']:+10.2f} {r['delta_tuned_pct']:+10.2f}")

    n_pass_uniform = sum(1 for r in rows if r["delta_uniform_pct"] > 0)
    n_pass_tuned = sum(1 for r in rows if r["delta_tuned_pct"] > 0)
    print(f"\nuniform   passes Gate 1 on {n_pass_uniform}/{len(rows)}")
    print(f"val-tuned passes Gate 1 on {n_pass_tuned}/{len(rows)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2, default=float)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    sys.exit(main())
