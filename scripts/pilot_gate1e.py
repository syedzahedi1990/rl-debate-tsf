"""Gate 1e: heterogeneous-panel premise check.

Panel:
  - arima                 (statistical anchor)
  - chronos_base          (foundation, T5 encoder-decoder)
  - chronos_tiny          (foundation, smaller variant)
  - moirai                (foundation, encoder-decoder with masked attention)
  - qwen_llmtime          (LLM, autoregressive sampling)

Five genuinely different mechanisms. Per-window forecasts are cached on disk
(distdeb.utils.cache), so a re-run after the first execution is I/O bound.

Per-dataset metrics for each agent + equal-weight ensemble + val-tuned
ensemble (simplex grid with 4 stratified val blocks, picks weights that
minimize mean per-block CRPS).

Headline question: does the val-tuned ensemble beat the best single agent
on test CRPS, on a majority of ETT datasets? If yes, the panel has enough
complementary information that RL orchestration is worth building.
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

from distdeb.eval.metrics import empirical_coverage, mae, mse, quantile_crps
from distdeb.utils.cache import ForecastCache, cached_forecast


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
    out = []
    n = 0
    for i in range(start_idx, end_idx, stride):
        if i - seq_len < 0:
            continue
        out.append((data[i - seq_len : i], data[i : i + pred_len]))
        n += 1
        if limit is not None and n >= limit:
            break
    return out


def stratified_val_blocks(series, seq_len, pred_len, n_blocks, block_size):
    val_lo = series["train_end"] + seq_len
    val_hi = series["val_end"] - pred_len + 1
    block_span = max(block_size + 1, (val_hi - val_lo) // n_blocks)
    return [
        _make_windows(series, val_lo + b * block_span, min(val_lo + b * block_span + block_size, val_hi),
                      seq_len, pred_len)
        for b in range(n_blocks)
    ]


def test_windows(series, seq_len, pred_len, stride, limit):
    start = series["val_end"]
    end = len(series["data"]) - pred_len + 1
    return _make_windows(series, start, end, seq_len, pred_len, stride, limit)


def _metrics(q, levels, targets):
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


def simplex_grid(A, resolution):
    def recur(remaining, depth, partial, out):
        if depth == 1:
            out.append(tuple(partial + [remaining]))
            return
        for i in range(remaining + 1):
            recur(remaining - i, depth - 1, partial + [i], out)

    out = []
    recur(resolution, A, [], out)
    return out


def best_weights_multiblock(agent_q_val_blocks, levels, val_tgt_blocks, resolution=8):
    A = agent_q_val_blocks[0].shape[0]
    grid = simplex_grid(A, resolution)
    best_w, best_score = None, float("inf")
    for tup in grid:
        w = np.array(tup, dtype=np.float64) / resolution
        scores = [_crps_of_mixture(w, q, levels, t) for q, t in zip(agent_q_val_blocks, val_tgt_blocks)]
        s = float(np.mean(scores))
        if s < best_score:
            best_score, best_w = s, w
    return best_w, best_score


def build_panel(args):
    """Construct only the agents in args.agents.

    Lazy construction lets us skip slow agents (Qwen, ARIMA) when they're
    not in the panel for a given run.
    """
    panel: Dict[str, object] = {}
    if "arima" in args.agents:
        from distdeb.agents.arima_agent import ARIMAAgent
        panel["arima"] = ARIMAAgent(order=tuple(int(x) for x in args.arima_order.split(",")))
    if "chronos_base" in args.agents:
        from distdeb.agents.chronos_agent import ChronosBoltAgent
        panel["chronos_base"] = ChronosBoltAgent(model_id="amazon/chronos-bolt-base")
    if "chronos_tiny" in args.agents:
        from distdeb.agents.chronos_agent import ChronosBoltAgent
        panel["chronos_tiny"] = ChronosBoltAgent(model_id="amazon/chronos-bolt-tiny")
    if "moirai" in args.agents:
        from distdeb.agents.moirai_agent import MoiraiAgent
        panel["moirai"] = MoiraiAgent(model_id="Salesforce/moirai-1.1-R-base", num_samples=args.moirai_samples)
    if "qwen_llmtime" in args.agents:
        from distdeb.agents.llmtime_agent import QwenLLMTimeAgent
        panel["qwen_llmtime"] = QwenLLMTimeAgent(
            model_id=args.qwen_model,
            n_samples=args.qwen_samples,
            temperature=args.qwen_temp,
        )
    return panel


def _agent_forecast(agent, name, cache, dataset, seq_len, horizon, split, histories, levels, batch_size):
    """Compute or load cached quantiles for one agent on one split."""
    if hasattr(agent, "forecast_batch"):
        compute = lambda h, H, lv: agent.forecast_batch(h, H, lv, batch_size=batch_size)
    else:
        # ARIMA path: per-window forecasts.
        def compute(h, H, lv):
            qs = [agent.forecast(h[i], H, lv).quantiles for i in range(h.shape[0])]
            return np.stack(qs).astype(np.float32)
    return cached_forecast(
        cache=cache,
        dataset=dataset,
        agent_name=name,
        seq_len=seq_len,
        horizon=horizon,
        split=split,
        histories=histories,
        levels=levels,
        compute_fn=compute,
    )


def run_dataset(ds_name, csv_path, freq, args, panel, levels, cache):
    if not os.path.exists(csv_path):
        return None
    print(f"\n--- {ds_name} ---")
    series = _load_ett(csv_path, freq)
    val_blocks = stratified_val_blocks(series, args.seq_len, args.horizon, args.n_val_blocks, args.val_windows)
    test_w = test_windows(series, args.seq_len, args.horizon, args.stride,
                          None if args.windows == -1 else args.windows)
    print(f"[data] val blocks: {[len(b) for b in val_blocks]}, test: {len(test_w)}")

    val_hist_blocks = [np.stack([h.squeeze(-1) for h, _ in b]).astype(np.float32) for b in val_blocks]
    val_tgt_blocks = [np.stack([t.squeeze(-1) for _, t in b]).astype(np.float32) for b in val_blocks]
    test_hist = np.stack([h.squeeze(-1) for h, _ in test_w]).astype(np.float32)
    test_tgt = np.stack([t.squeeze(-1) for _, t in test_w]).astype(np.float32)

    test_q = {}
    val_q_blocks: List[np.ndarray] = [None] * args.n_val_blocks  # type: ignore
    timings = {}
    for name, agent in panel.items():
        bs = args.batch_size.get(name, 64) if isinstance(args.batch_size, dict) else args.batch_size
        print(f"[agent] {name} (bs={bs})...")
        t0 = time.perf_counter()
        test_q[name] = _agent_forecast(agent, name, cache, ds_name, args.seq_len, args.horizon, "test",
                                       test_hist, levels, batch_size=bs)
        for bi, vh in enumerate(val_hist_blocks):
            vq = _agent_forecast(agent, name, cache, ds_name, args.seq_len, args.horizon, f"val{bi}",
                                 vh, levels, batch_size=bs)
            if val_q_blocks[bi] is None:
                val_q_blocks[bi] = vq[None]
            else:
                val_q_blocks[bi] = np.concatenate([val_q_blocks[bi], vq[None]], axis=0)
        timings[name] = time.perf_counter() - t0
        print(f"  {timings[name]:.1f}s")

    names = list(panel.keys())
    test_stack = np.stack([test_q[n] for n in names], axis=0)
    per_agent = {n: _metrics(test_q[n], levels, test_tgt) | {"wall_s": timings[n]} for n in names}

    uniform_q = np.mean(test_stack, axis=0)
    uniform_m = _metrics(uniform_q, levels, test_tgt)

    best_w, val_mean_crps = best_weights_multiblock(val_q_blocks, levels, val_tgt_blocks,
                                                    resolution=args.grid_resolution)
    tuned_q = np.einsum("a,anqh->nqh", best_w, test_stack)
    tuned_m = _metrics(tuned_q, levels, test_tgt)

    best_single = min(names, key=lambda n: per_agent[n]["CRPS"])
    bs_crps = per_agent[best_single]["CRPS"]
    d_uniform = 100 * (bs_crps - uniform_m["CRPS"]) / bs_crps
    d_tuned = 100 * (bs_crps - tuned_m["CRPS"]) / bs_crps

    print(f"\n  val-best weights ({args.n_val_blocks}-block mean CRPS={val_mean_crps:.4f}):")
    for n, w in zip(names, best_w):
        print(f"    {n:20s} {float(w):.2f}")
    print(f"  {'method':25s} {'MSE':>8s} {'MAE':>8s} {'CRPS':>8s} {'cov80':>8s}")
    for n in names:
        m = per_agent[n]
        print(f"  {n:25s} {m['MSE']:8.4f} {m['MAE']:8.4f} {m['CRPS']:8.4f} {m['coverage_80']:8.4f}")
    print(f"  {'uniform':25s} {uniform_m['MSE']:8.4f} {uniform_m['MAE']:8.4f} {uniform_m['CRPS']:8.4f} {uniform_m['coverage_80']:8.4f}")
    print(f"  {'val-tuned':25s} {tuned_m['MSE']:8.4f} {tuned_m['MAE']:8.4f} {tuned_m['CRPS']:8.4f} {tuned_m['coverage_80']:8.4f}")
    print(f"  -> uniform   vs best({best_single}): {d_uniform:+.2f}%  {'PASS' if d_uniform > 0 else 'FAIL'}")
    print(f"  -> val-tuned vs best({best_single}): {d_tuned:+.2f}%  {'PASS' if d_tuned > 0 else 'FAIL'}")

    return {
        "dataset": ds_name,
        "horizon": args.horizon,
        "agents": names,
        "val_best_weights": [float(x) for x in best_w],
        "per_agent_test": per_agent,
        "uniform_test": uniform_m,
        "val_tuned_test": tuned_m,
        "best_single": best_single,
        "delta_uniform_pct": float(d_uniform),
        "delta_tuned_pct": float(d_tuned),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--windows", type=int, default=256)
    p.add_argument("--val-windows", type=int, default=64)
    p.add_argument("--n-val-blocks", type=int, default=4)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument(
        "--agents",
        nargs="+",
        default=["arima", "chronos_base", "chronos_tiny", "moirai", "qwen_llmtime"],
    )
    p.add_argument("--qwen-model", default="Qwen/Qwen2.5-1.5B", help="Use Qwen/Qwen2.5-7B for the paper run.")
    p.add_argument("--qwen-samples", type=int, default=20)
    p.add_argument("--qwen-temp", type=float, default=0.7)
    p.add_argument("--qwen-batch", type=int, default=4)
    p.add_argument("--moirai-samples", type=int, default=100)
    p.add_argument("--moirai-batch", type=int, default=32)
    p.add_argument("--chronos-batch", type=int, default=128)
    p.add_argument("--arima-order", default="2,1,1")
    p.add_argument("--grid-resolution", type=int, default=8)
    p.add_argument("--cache-root", default="data_cache/forecasts")
    p.add_argument("--out", default="results/pilot_gate1e.json")
    args = p.parse_args()

    # Per-agent batch sizes (Qwen is the bottleneck; Moirai is medium; Chronos is huge).
    args.batch_size = {
        "qwen_llmtime": args.qwen_batch,
        "moirai": args.moirai_batch,
        "chronos_base": args.chronos_batch,
        "chronos_tiny": args.chronos_batch,
        "arima": 1,
    }

    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    print(f"[setup] panel: {args.agents}")
    print(f"[setup] H={args.horizon}, windows={args.windows}, val={args.n_val_blocks}x{args.val_windows}")
    panel = build_panel(args)
    cache = ForecastCache(root=args.cache_root)

    rows = []
    for n, p_, f in DATASETS:
        r = run_dataset(n, p_, f, args, panel, levels, cache)
        if r:
            rows.append(r)

    print("\n\n=== Gate 1e summary ===")
    print(f"{'dataset':10s} {'best_single':15s} {'uniform%':>10s} {'tuned%':>10s}")
    for r in rows:
        print(f"{r['dataset']:10s} {r['best_single']:15s} {r['delta_uniform_pct']:+10.2f} {r['delta_tuned_pct']:+10.2f}")
    n_pass_u = sum(1 for r in rows if r["delta_uniform_pct"] > 0)
    n_pass_t = sum(1 for r in rows if r["delta_tuned_pct"] > 0)
    print(f"\nuniform   passes Gate 1 on {n_pass_u}/{len(rows)}")
    print(f"val-tuned passes Gate 1 on {n_pass_t}/{len(rows)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": {k: v for k, v in vars(args).items() if k != "batch_size"}, "rows": rows},
                  f, indent=2, default=float)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    sys.exit(main())
