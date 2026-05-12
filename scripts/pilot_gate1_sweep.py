"""Gate 1 robustness sweep across the ETT-small suite.

Runs pilot_gate1's panel (ARIMA + Chronos-Bolt + equal-weight ensemble) on
ETTh1, ETTh2, ETTm1, ETTm2 at H=96 and reports a results table. Used to
sanity-check that the Gate-1 calibration story from ETTh1 isn't dataset
luck.

ETTm1/m2 have ~4x more test windows than ETTh*, so ARIMA there is slow.
Use --skip-arima for a fast Chronos-only sweep, or --windows N to subsample.

Usage:
  python scripts/pilot_gate1_sweep.py --windows 1024 --horizon 96
  python scripts/pilot_gate1_sweep.py --windows -1 --horizon 96 --skip-arima
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distdeb.agents.arima_agent import ARIMAAgent
from distdeb.agents.chronos_agent import ChronosBoltAgent
from distdeb.data.loaders import iter_test_windows, load_etth1_windows
from distdeb.eval.metrics import empirical_coverage, mae, mse, quantile_crps


DATASETS = [
    ("ETTh1", "dataset/ETT-small/ETTh1.csv", "hour"),
    ("ETTh2", "dataset/ETT-small/ETTh2.csv", "hour"),
    ("ETTm1", "dataset/ETT-small/ETTm1.csv", "minute"),
    ("ETTm2", "dataset/ETT-small/ETTm2.csv", "minute"),
]


def _metrics(quantiles, levels, targets):
    median_idx = int(np.argmin(np.abs(levels - 0.5)))
    median = quantiles[:, median_idx, :]
    q_for_crps = np.transpose(quantiles, (1, 0, 2))
    return {
        "MSE": mse(median, targets),
        "MAE": mae(median, targets),
        "CRPS": quantile_crps(q_for_crps, levels, targets),
        "coverage_80": empirical_coverage(q_for_crps, levels, targets, alpha=0.8),
    }


def run_one(name, csv_path, freq, args, chronos, levels):
    if not os.path.exists(csv_path):
        print(f"[skip] {csv_path} not found")
        return None
    print(f"\n--- {name} ---")
    # load_etth1_windows is freq-agnostic for the CSV reading; the freq
    # only matters for the train/val split boundaries hard-coded inside.
    # We patch the boundaries by inlining minute-frequency support here:
    series = _load_ett(csv_path, freq, features="S")
    print(f"[data] shape={series['data'].shape}, train_end={series['train_end']}, val_end={series['val_end']}")

    windows = list(_iter_windows(series, args.seq_len, args.horizon, args.stride, args.windows))
    print(f"[data] {len(windows)} test windows")

    histories = np.stack([h.squeeze(-1) for h, _ in windows]).astype(np.float32)
    targets = np.stack([t.squeeze(-1) for _, t in windows]).astype(np.float32)

    agents_q = {}
    timings = {}

    if not args.skip_arima:
        print("[agent] ARIMA...")
        arima = ARIMAAgent(order=tuple(int(x) for x in args.arima_order.split(",")))
        t0 = time.perf_counter()
        agents_q["arima"] = np.stack(
            [arima.forecast(h, args.horizon, levels).quantiles for h in histories]
        ).astype(np.float32)
        timings["arima"] = time.perf_counter() - t0
        print(f"  {timings['arima']:.1f}s")

    print("[agent] chronos-bolt...")
    t0 = time.perf_counter()
    agents_q["chronos"] = chronos.forecast_batch(histories, args.horizon, levels, batch_size=args.chronos_batch)
    timings["chronos"] = time.perf_counter() - t0
    print(f"  {timings['chronos']:.1f}s")

    row = {"dataset": name, "n_windows": len(windows), "metrics": {}}
    for n, q in agents_q.items():
        row["metrics"][n] = _metrics(q, levels, targets) | {"wall_s": timings[n]}
    if len(agents_q) >= 2:
        ens = np.mean(np.stack(list(agents_q.values()), axis=0), axis=0)
        row["metrics"]["equal_weight_ensemble"] = _metrics(ens, levels, targets)
    return row


def _ett_boundaries(freq):
    if freq == "hour":
        return 12 * 30 * 24, 16 * 30 * 24
    if freq == "minute":
        return 12 * 30 * 24 * 4, 16 * 30 * 24 * 4
    raise ValueError(freq)


def _load_ett(csv_path, freq, features="S", target="OT"):
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    arr = df[[target]].values.astype(np.float32) if features == "S" else df.values.astype(np.float32)
    train_end, val_end = _ett_boundaries(freq)
    mean = arr[:train_end].mean(axis=0)
    std = arr[:train_end].std(axis=0) + 1e-8
    return {"data": (arr - mean) / std, "train_end": train_end, "val_end": val_end, "mean": mean, "std": std}


def _iter_windows(series, seq_len, pred_len, stride, limit):
    data = series["data"]
    test_start = series["val_end"]
    end = len(data) - pred_len
    n = 0
    lim = None if limit == -1 else limit
    for i in range(test_start, end, stride):
        if i - seq_len < 0:
            continue
        yield data[i - seq_len : i], data[i : i + pred_len]
        n += 1
        if lim is not None and n >= lim:
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--windows", type=int, default=1024, help="Per-dataset cap; -1 for all.")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--chronos-model", default="amazon/chronos-bolt-base")
    p.add_argument("--chronos-batch", type=int, default=128)
    p.add_argument("--arima-order", default="2,1,1")
    p.add_argument("--skip-arima", action="store_true")
    p.add_argument("--out", default="results/pilot_gate1_sweep.json")
    args = p.parse_args()

    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    print(f"[setup] loading {args.chronos_model} once and reusing across datasets")
    chronos = ChronosBoltAgent(model_id=args.chronos_model)

    rows = []
    for name, csv_path, freq in DATASETS:
        rows.append(run_one(name, csv_path, freq, args, chronos, levels))

    print("\n\n=== Gate 1 sweep summary ===")
    print(f"{'dataset':10s} {'agent':25s} {'MSE':>8s} {'MAE':>8s} {'CRPS':>8s} {'cov80':>8s}")
    n_pass = 0
    n_compared = 0
    for row in rows:
        if row is None:
            continue
        for agent_name, m in row["metrics"].items():
            print(f"{row['dataset']:10s} {agent_name:25s} {m['MSE']:8.4f} {m['MAE']:8.4f} {m['CRPS']:8.4f} {m['coverage_80']:8.4f}")
        # Per-dataset verdict
        singles = {n: m["CRPS"] for n, m in row["metrics"].items() if n != "equal_weight_ensemble"}
        if "equal_weight_ensemble" in row["metrics"] and singles:
            best_n = min(singles, key=singles.get)
            ens = row["metrics"]["equal_weight_ensemble"]["CRPS"]
            delta = 100 * (singles[best_n] - ens) / singles[best_n]
            verdict = "PASS" if ens < singles[best_n] else "FAIL"
            print(f"{row['dataset']:10s} -> ensemble vs best({best_n}): {delta:+.2f}%  {verdict}")
            n_compared += 1
            if ens < singles[best_n]:
                n_pass += 1

    if n_compared:
        print(f"\n=== overall: ensemble beats best-single on {n_pass}/{n_compared} datasets ===")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2, default=float)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    sys.exit(main())
