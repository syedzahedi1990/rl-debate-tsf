"""Conformal-calibrated single-agent baseline.

Answers the inevitable reviewer question: "why not just apply split-conformal
prediction to the best single agent?" Compares:

  - Naive (uncalibrated) best single agent
  - Split-conformal calibrated best single agent (uses val for calibration)
  - Uniform ensemble
  - RL orchestrator (if results/orchestrator.pt + results/pilot_gate2.json exist)

Reports CRPS, coverage_80, mean_n_calls per dataset.

Usage:
  python scripts/diag_conformal.py --windows 256
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distdeb.baselines.conformal import split_conformal_calibrate
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
    return {"data": (arr - mean) / std, "train_end": train_end, "val_end": val_end}


def _make_windows(series, start_idx, end_idx, seq_len, pred_len, limit=None):
    data = series["data"]
    out = []
    for i in range(start_idx, end_idx):
        if i - seq_len < 0:
            continue
        out.append((data[i - seq_len : i], data[i : i + pred_len]))
        if limit is not None and len(out) == limit:
            break
    return out


def val_targets(series, seq_len, pred_len, n_blocks=4, block_size=64):
    """Concatenate the 4 val blocks the cache stored under split=val0..val3."""
    lo = series["train_end"] + seq_len
    hi = series["val_end"] - pred_len + 1
    block_span = max(block_size + 1, (hi - lo) // n_blocks)
    out = []
    for b in range(n_blocks):
        start = lo + b * block_span
        end = min(start + block_size, hi)
        for w in _make_windows(series, start, end, seq_len, pred_len):
            out.append(w[1].squeeze(-1))
    return np.stack(out).astype(np.float32)


def test_targets(series, seq_len, pred_len, n):
    out = []
    for h, t in _make_windows(series, series["val_end"], len(series["data"]) - pred_len + 1, seq_len, pred_len, limit=n):
        out.append(t.squeeze(-1))
    return np.stack(out).astype(np.float32)


def _metrics(q, levels, t):
    median = q[:, int(np.argmin(np.abs(levels - 0.5))), :]
    qfc = np.transpose(q, (1, 0, 2))
    return {
        "MSE": float(mse(median, t)),
        "MAE": float(mae(median, t)),
        "CRPS": float(quantile_crps(qfc, levels, t)),
        "coverage_80": float(empirical_coverage(qfc, levels, t, alpha=0.8)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--windows", type=int, default=256)
    p.add_argument("--val-block-size", type=int, default=64)
    p.add_argument("--cache-root", default=None)
    p.add_argument("--agents", nargs="+", default=["arima", "chronos_base", "chronos_tiny", "moirai"])
    p.add_argument("--gate2-results", default="results/pilot_gate2.json",
                   help="RL eval JSON to include for comparison (optional).")
    p.add_argument("--out", default="results/diag_conformal.json")
    args = p.parse_args()

    cache = ForecastCache(root=args.cache_root)
    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

    rl_results: Dict[str, dict] = {}
    if os.path.exists(args.gate2_results):
        with open(args.gate2_results) as f:
            gate2 = json.load(f)
        rl_results = gate2.get("per_dataset", {})

    summary = {}
    for ds_id, (name, csv_path, freq) in enumerate(DATASETS):
        if not os.path.exists(csv_path):
            continue
        series = _load_ett(csv_path, freq)
        val_tgt = val_targets(series, args.seq_len, args.horizon, n_blocks=4, block_size=args.val_block_size)
        test_tgt = test_targets(series, args.seq_len, args.horizon, args.windows)

        # Load val + test forecasts per agent.
        val_q, test_q = {}, {}
        for a in args.agents:
            # Val was stored split-wise (val0..val3); concatenate.
            v = None
            for bi in range(4):
                part = cache.load(
                    dataset=name, agent=a, seq_len=args.seq_len, horizon=args.horizon,
                    split=f"val{bi}", n_windows=args.val_block_size, levels=levels,
                )
                if part is None:
                    v = None
                    break
                v = part if v is None else np.concatenate([v, part], axis=0)
            t = cache.load(
                dataset=name, agent=a, seq_len=args.seq_len, horizon=args.horizon, split="test",
                n_windows=args.windows, levels=levels,
            )
            if v is None or t is None:
                print(f"  [miss] {name}/{a}; skipping")
                val_q = {}
                break
            val_q[a] = v[: len(val_tgt)]
            test_q[a] = t
        if not val_q:
            continue

        # Per-agent metrics (naive + conformal).
        naive = {a: _metrics(test_q[a], levels, test_tgt) for a in args.agents}
        conformal = {}
        for a in args.agents:
            cal, c_eps = split_conformal_calibrate(val_q[a], val_tgt, test_q[a], levels, target_coverage=0.8)
            m = _metrics(cal, levels, test_tgt)
            m["epsilon"] = float(c_eps)
            conformal[a] = m

        # Uniform ensemble (naive + conformal).
        stack_test = np.stack([test_q[a] for a in args.agents], axis=0).mean(axis=0)
        stack_val = np.stack([val_q[a] for a in args.agents], axis=0).mean(axis=0)
        uniform_naive = _metrics(stack_test, levels, test_tgt)
        cal_unif, c_unif = split_conformal_calibrate(stack_val, val_tgt, stack_test, levels, target_coverage=0.8)
        uniform_cal = _metrics(cal_unif, levels, test_tgt)
        uniform_cal["epsilon"] = float(c_unif)

        # Headline numbers.
        best_naive = min(naive, key=lambda a: naive[a]["CRPS"])
        best_cal = min(conformal, key=lambda a: conformal[a]["CRPS"])
        print(f"\n=== {name} ===")
        print(f"{'method':30s} {'CRPS':>8s} {'cov80':>8s} {'eps':>8s}")
        for a in args.agents:
            m = naive[a]
            print(f"  naive {a:24s} {m['CRPS']:8.4f} {m['coverage_80']:8.4f}")
            mc = conformal[a]
            print(f"  conf  {a:24s} {mc['CRPS']:8.4f} {mc['coverage_80']:8.4f} {mc['epsilon']:8.4f}")
        print(f"  uniform (naive)              {uniform_naive['CRPS']:8.4f} {uniform_naive['coverage_80']:8.4f}")
        print(f"  uniform (conformal)          {uniform_cal['CRPS']:8.4f} {uniform_cal['coverage_80']:8.4f} {uniform_cal['epsilon']:8.4f}")
        if name in rl_results:
            rl = rl_results[name]["rl"]
            print(f"  rl_orchestrator              {rl['CRPS']:8.4f} {rl['coverage_80']:8.4f}  (n_calls={rl.get('mean_n_calls', 0):.2f})")

        summary[name] = {
            "best_naive_agent": best_naive,
            "best_cal_agent": best_cal,
            "naive": naive,
            "conformal": conformal,
            "uniform_naive": uniform_naive,
            "uniform_conformal": uniform_cal,
            "rl": rl_results.get(name, {}).get("rl"),
        }

    # Aggregate verdict.
    print("\n=== Conformal-vs-RL aggregate ===")
    for ds_name, s in summary.items():
        ba = s["best_naive_agent"]
        bc = s["best_cal_agent"]
        naive_best_crps = s["naive"][ba]["CRPS"]
        cal_best_crps = s["conformal"][bc]["CRPS"]
        cal_best_cov = s["conformal"][bc]["coverage_80"]
        line = f"{ds_name:10s}  best naive: {ba} (CRPS={naive_best_crps:.4f})  " \
               f"best conformal: {bc} (CRPS={cal_best_crps:.4f}, cov={cal_best_cov:.3f})"
        if s["rl"]:
            line += f"  RL CRPS={s['rl']['CRPS']:.4f} cov={s['rl']['coverage_80']:.3f}"
        print(line)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "config": vars(args)}, f, indent=2, default=float)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    sys.exit(main())
