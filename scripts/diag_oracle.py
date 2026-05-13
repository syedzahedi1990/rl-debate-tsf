"""Diagnostic: per-window oracle bound for cached forecasts.

Reads all agent forecasts cached by pilot_gate1e and computes:

  - Per-agent test CRPS (sanity-check the Gate 1e numbers).
  - Per-window CRPS for each agent.
  - Oracle CRPS: average of `min_a CRPS(a, w)` over windows w.
      This is the upper bound a "perfect per-window router" achieves.
      Any real orchestrator that doesn't see the target must do worse.
  - Distribution of oracle's per-window agent choice. If oracle chooses
    different agents on different windows, the panel has *latent*
    decorrelation worth exploiting. If it collapses to one agent, no
    routing can help.
  - Uniform ensemble CRPS (all agents and "minus Qwen" variants), for
    headline comparison.

Verdict thresholds:
  - oracle gap >= +5% vs best single -> routing has real headroom.
  - oracle gap < +5% -> the panel is too correlated; pivot the framing.

Usage:
  python scripts/diag_oracle.py --windows 256 --horizon 96
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distdeb.eval.metrics import empirical_coverage, mae, mse, quantile_crps
from distdeb.utils.cache import ForecastCache


DATASETS = [
    ("ETTh1", "dataset/ETT-small/ETTh1.csv", "hour"),
    ("ETTh2", "dataset/ETT-small/ETTh2.csv", "hour"),
    ("ETTm1", "dataset/ETT-small/ETTm1.csv", "minute"),
    ("ETTm2", "dataset/ETT-small/ETTm2.csv", "minute"),
]


def _ett_bounds(freq):
    if freq == "hour":
        return 12 * 30 * 24, 16 * 30 * 24
    return 12 * 30 * 24 * 4, 16 * 30 * 24 * 4


def _load_ett(csv_path, freq, target="OT"):
    import pandas as pd
    df = pd.read_csv(csv_path).drop(columns=["date"], errors="ignore")
    arr = df[[target]].values.astype(np.float32)
    train_end, val_end = _ett_bounds(freq)
    mean = arr[:train_end].mean(axis=0)
    std = arr[:train_end].std(axis=0) + 1e-8
    return {"data": (arr - mean) / std, "val_end": val_end}


def test_targets(series, seq_len, pred_len, n):
    data = series["data"]
    start = series["val_end"]
    end = len(data) - pred_len + 1
    tgts = []
    for i in range(start, end):
        if i - seq_len < 0:
            continue
        tgts.append(data[i : i + pred_len].squeeze(-1))
        if len(tgts) == n:
            break
    return np.stack(tgts).astype(np.float32)


def per_window_crps(quantiles: np.ndarray, levels: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """quantiles (N, Q, H), targets (N, H). Returns (N,) — CRPS per window."""
    losses = []
    for qi, lv in enumerate(levels):
        diff = targets - quantiles[:, qi, :]
        pinball = np.maximum(lv * diff, (lv - 1) * diff)
        losses.append(np.mean(pinball, axis=1))  # mean over horizon -> (N,)
    stacked = np.stack(losses, axis=0)  # (Q, N)
    return 2.0 * np.mean(stacked, axis=0)


def _ensemble_metrics(stack: np.ndarray, levels: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    avg = np.mean(stack, axis=0)
    return {
        "CRPS": quantile_crps(np.transpose(avg, (1, 0, 2)), levels, targets),
        "coverage_80": empirical_coverage(np.transpose(avg, (1, 0, 2)), levels, targets, alpha=0.8),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--windows", type=int, default=256)
    p.add_argument("--cache-root", default="data_cache/forecasts")
    p.add_argument(
        "--agents",
        nargs="+",
        default=["arima", "chronos_base", "chronos_tiny", "moirai", "qwen_llmtime"],
    )
    p.add_argument("--out", default="results/diag_oracle.json")
    args = p.parse_args()

    cache = ForecastCache(root=args.cache_root)
    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

    rows = []
    for ds_name, csv_path, freq in DATASETS:
        if not os.path.exists(csv_path):
            print(f"[skip] {csv_path}")
            continue
        series = _load_ett(csv_path, freq)
        targets = test_targets(series, args.seq_len, args.horizon, args.windows)

        agent_q: Dict[str, np.ndarray] = {}
        for ag_name in args.agents:
            q = cache.load(
                dataset=ds_name,
                agent=ag_name,
                seq_len=args.seq_len,
                horizon=args.horizon,
                split="test",
                n_windows=args.windows,
                levels=levels,
            )
            if q is None:
                print(f"  [miss] {ds_name} / {ag_name}")
                continue
            agent_q[ag_name] = q

        if len(agent_q) < 2:
            continue

        # Per-window CRPS for each agent.
        per_w = {n: per_window_crps(q, levels, targets) for n, q in agent_q.items()}
        per_agent_crps = {n: float(np.mean(c)) for n, c in per_w.items()}
        per_agent_cov = {
            n: empirical_coverage(np.transpose(q, (1, 0, 2)), levels, targets, alpha=0.8)
            for n, q in agent_q.items()
        }

        # Oracle.
        names = list(per_w.keys())
        crps_matrix = np.stack([per_w[n] for n in names], axis=0)  # (A, N)
        oracle_per_window = np.min(crps_matrix, axis=0)
        oracle_crps = float(np.mean(oracle_per_window))
        oracle_choices = np.argmin(crps_matrix, axis=0)
        choice_dist = np.bincount(oracle_choices, minlength=len(names))

        # Uniform ensembles.
        stack_all = np.stack([agent_q[n] for n in names], axis=0)
        uniform_all = _ensemble_metrics(stack_all, levels, targets)

        no_qwen = [n for n in names if n != "qwen_llmtime"]
        stack_nq = np.stack([agent_q[n] for n in no_qwen], axis=0)
        uniform_nq = _ensemble_metrics(stack_nq, levels, targets) if len(no_qwen) >= 2 else None

        # Print
        best_single = min(per_agent_crps, key=per_agent_crps.get)
        bs_crps = per_agent_crps[best_single]
        print(f"\n=== {ds_name} (n={args.windows}, H={args.horizon}) ===")
        print(f"{'agent':22s} {'CRPS':>8s} {'cov80':>8s}")
        for n in names:
            print(f"  {n:20s} {per_agent_crps[n]:8.4f} {per_agent_cov[n]:8.4f}")
        print(f"\nBest single: {best_single} (CRPS={bs_crps:.4f})")
        print(f"Uniform (all):       CRPS={uniform_all['CRPS']:.4f}  cov80={uniform_all['coverage_80']:.4f}  "
              f"({100*(bs_crps-uniform_all['CRPS'])/bs_crps:+.2f}% vs best single)")
        if uniform_nq:
            print(f"Uniform (no Qwen):   CRPS={uniform_nq['CRPS']:.4f}  cov80={uniform_nq['coverage_80']:.4f}  "
                  f"({100*(bs_crps-uniform_nq['CRPS'])/bs_crps:+.2f}%)")
        print(f"Oracle per-window:   CRPS={oracle_crps:.4f}  "
              f"({100*(bs_crps-oracle_crps)/bs_crps:+.2f}% headroom for a perfect router)")
        print("Oracle agent picks:")
        for i, n in enumerate(names):
            pct = 100 * choice_dist[i] / len(targets)
            print(f"  {n:20s} {choice_dist[i]:4d} ({pct:.1f}%)")

        rows.append({
            "dataset": ds_name,
            "best_single": best_single,
            "best_single_crps": bs_crps,
            "per_agent_crps": per_agent_crps,
            "per_agent_cov80": per_agent_cov,
            "uniform_all_crps": uniform_all["CRPS"],
            "uniform_all_cov80": uniform_all["coverage_80"],
            "uniform_no_qwen_crps": uniform_nq["CRPS"] if uniform_nq else None,
            "uniform_no_qwen_cov80": uniform_nq["coverage_80"] if uniform_nq else None,
            "oracle_crps": oracle_crps,
            "oracle_headroom_pct": float(100 * (bs_crps - oracle_crps) / bs_crps),
            "oracle_choice_distribution": {n: int(c) for n, c in zip(names, choice_dist)},
        })

    print("\n\n=== Oracle headroom summary ===")
    print(f"{'dataset':10s} {'best_single':16s} {'best_crps':>10s} {'oracle_crps':>12s} {'headroom%':>10s}")
    for r in rows:
        print(f"{r['dataset']:10s} {r['best_single']:16s} {r['best_single_crps']:10.4f} "
              f"{r['oracle_crps']:12.4f} {r['oracle_headroom_pct']:+10.2f}")
    mean_headroom = float(np.mean([r["oracle_headroom_pct"] for r in rows]))
    print(f"\nMean oracle headroom across datasets: {mean_headroom:+.2f}%")
    print("Interpretation:")
    print("  >= 10%  -> routing has clear potential; build RL.")
    print("   5-10%  -> routing has modest potential; cautious build.")
    print("    < 5%  -> panel too correlated; pivot framing.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2, default=float)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    sys.exit(main())
