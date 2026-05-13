"""Path C diagnostic: same-distribution training and evaluation.

Splits each dataset's TEST cache 80/20 (randomized within-dataset). Trains
one PPO policy on the 80% portions mixed across datasets, evaluates on the
20% holdouts per-dataset.

Hypothesis: Gate 2's val/test distribution shift may be the bottleneck.
If matched-distribution training extracts most of the oracle headroom, we
know the method works in principle and we should push for top-tier (Path B
in the plan). If even matched training gives marginal gain, the method has
a ceiling at H=96 on ETT (Path A or pivot).

This is INVALID as a paper result (training on test) but it's a clean
diagnostic for whether the method itself can extract per-window decorrelation.

Usage:
  python scripts/diag_within_dataset.py --n-iters 500 --windows 256
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from distdeb.env.refinement_env import RefinementEnv
from distdeb.eval.metrics import empirical_coverage, mae, mse, quantile_crps
from distdeb.orchestrator.policy import PolicyValueNet
from distdeb.orchestrator.ppo import PPOConfig, PPOTrainer
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


def test_windows(series, seq_len, pred_len, n):
    data = series["data"]
    start = series["val_end"]
    end = len(data) - pred_len + 1
    out = []
    for i in range(start, end):
        if i - seq_len < 0:
            continue
        out.append((data[i - seq_len : i], data[i : i + pred_len]))
        if len(out) == n:
            break
    return out


def _metrics_from_q(q, levels, t):
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
    p.add_argument("--windows", type=int, default=256, help="Total test cache windows per dataset.")
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--cache-root", default=None)
    p.add_argument("--agents", nargs="+", default=["arima", "chronos_base", "chronos_tiny", "moirai"])
    p.add_argument("--cost-weight", type=float, default=0.001)
    p.add_argument("--entropy-coef", type=float, default=0.05)
    p.add_argument("--n-iters", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--include-dataset-id", action="store_true", default=True)
    p.add_argument("--no-dataset-id", dest="include_dataset_id", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/diag_within_dataset.json")
    args = p.parse_args()

    cache = ForecastCache(root=args.cache_root)
    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    rng = np.random.default_rng(args.seed)

    n_train_per_ds = int(args.windows * args.train_fraction)
    n_eval_per_ds = args.windows - n_train_per_ds
    print(f"[setup] within-dataset split: {n_train_per_ds} train / {n_eval_per_ds} eval per dataset")

    train_hist_list, train_tgt_list, train_id_list = [], [], []
    train_q_list = {a: [] for a in args.agents}
    eval_data = {}

    for ds_id, (name, csv_path, freq) in enumerate(DATASETS):
        if not os.path.exists(csv_path):
            print(f"[skip] {csv_path}")
            continue
        series = _load_ett(csv_path, freq)
        wins = test_windows(series, args.seq_len, args.horizon, args.windows)
        if len(wins) < args.windows:
            print(f"[warn] {name}: only {len(wins)} windows available; need {args.windows}")
        n = len(wins)
        hist = np.stack([h.squeeze(-1) for h, _ in wins]).astype(np.float32)
        tgt = np.stack([t.squeeze(-1) for _, t in wins]).astype(np.float32)

        # Load each agent's cached forecasts for the test split.
        ds_q = {}
        missing = False
        for a in args.agents:
            q = cache.load(
                dataset=name, agent=a, seq_len=args.seq_len, horizon=args.horizon,
                split="test", n_windows=n, levels=levels,
            )
            if q is None:
                print(f"  [miss] {name}/{a} — run pilot_gate1e first")
                missing = True
                break
            ds_q[a] = q
        if missing:
            continue

        # Random 80/20 split.
        idx = rng.permutation(n)
        n_train = int(n * args.train_fraction)
        train_idx = idx[:n_train]
        eval_idx = idx[n_train:]

        train_hist_list.append(hist[train_idx])
        train_tgt_list.append(tgt[train_idx])
        train_id_list.append(np.full(len(train_idx), ds_id, dtype=np.int32))
        for a in args.agents:
            train_q_list[a].append(ds_q[a][train_idx])
        eval_data[name] = {
            "ds_id": ds_id,
            "hist": hist[eval_idx],
            "tgt": tgt[eval_idx],
            "q": {a: ds_q[a][eval_idx] for a in args.agents},
        }

    if not train_hist_list:
        print("[err] no datasets had complete cache; run pilot_gate1e first")
        return 1

    train_hist = np.concatenate(train_hist_list, axis=0)
    train_tgt = np.concatenate(train_tgt_list, axis=0)
    train_ids = np.concatenate(train_id_list, axis=0)
    train_q = {a: np.concatenate(train_q_list[a], axis=0) for a in args.agents}
    n_datasets = int(train_ids.max()) + 1
    print(f"[setup] training mix: {len(train_hist)} windows across {n_datasets} datasets")

    # Per-agent training-CRPS sanity print.
    print("\n[diag] per-agent training-data CRPS:")
    for a in args.agents:
        losses = []
        for qi, lv in enumerate(levels):
            diff = train_tgt - train_q[a][:, qi, :]
            pinball = np.maximum(lv * diff, (lv - 1) * diff)
            losses.append(np.mean(pinball, axis=1))
        per_w = 2.0 * np.mean(np.stack(losses, axis=0), axis=0)
        print(f"  {a:20s} mean={per_w.mean():.4f}  std={per_w.std():.4f}")

    train_env = RefinementEnv(
        histories=train_hist,
        targets=train_tgt,
        agent_quantiles=train_q,
        levels=levels,
        dataset_ids=train_ids,
        n_datasets=n_datasets,
        include_dataset_id=args.include_dataset_id,
        cost_weight=args.cost_weight,
    )
    print(f"[setup] state_dim={train_env.state_dim}, n_actions={train_env.n_actions}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    policy = PolicyValueNet(train_env.state_dim, train_env.n_actions, hidden=128)
    cfg = PPOConfig(
        n_iters=args.n_iters,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        seed=args.seed,
    )
    trainer = PPOTrainer(train_env, policy, cfg, device=device)

    print(f"\n[train] {cfg.n_iters} PPO iters")
    history = trainer.train(log_every=max(1, cfg.n_iters // 10))

    # Eval per-dataset
    print("\n[eval] per-dataset (held-out 20% of test cache)")
    per_dataset = {}
    rl_deltas = []
    for ds_name, ed in eval_data.items():
        n_eval = len(ed["hist"])
        eval_env = RefinementEnv(
            histories=ed["hist"],
            targets=ed["tgt"],
            agent_quantiles=ed["q"],
            levels=levels,
            dataset_ids=np.full(n_eval, ed["ds_id"], dtype=np.int32),
            n_datasets=n_datasets,
            include_dataset_id=args.include_dataset_id,
            cost_weight=args.cost_weight,
        )
        result = trainer.evaluate(eval_env, n_windows=n_eval, deterministic=True)
        rl_metrics = _metrics_from_q(result["ensembles"], levels, ed["tgt"])

        per_agent = {a: _metrics_from_q(ed["q"][a], levels, ed["tgt"]) for a in args.agents}
        stack_all = np.stack([ed["q"][a] for a in args.agents], axis=0)
        uniform_q = np.mean(stack_all, axis=0)
        uniform_m = _metrics_from_q(uniform_q, levels, ed["tgt"])

        best_single = min(per_agent, key=lambda a: per_agent[a]["CRPS"])
        bs_crps = per_agent[best_single]["CRPS"]
        rl_delta = 100 * (bs_crps - rl_metrics["CRPS"]) / bs_crps
        rl_vs_unif = 100 * (uniform_m["CRPS"] - rl_metrics["CRPS"]) / uniform_m["CRPS"]
        rl_deltas.append(rl_delta)

        all_chosen = [c for cc in result["chosen"] for c in cc]
        choice_dist = np.bincount(all_chosen, minlength=len(args.agents))

        per_dataset[ds_name] = {
            "n_eval": n_eval,
            "per_agent": per_agent,
            "uniform": uniform_m,
            "rl": rl_metrics | {"mean_n_calls": result["mean_n_calls"]},
            "best_single": best_single,
            "rl_vs_best_crps_pct": float(rl_delta),
            "rl_vs_uniform_crps_pct": float(rl_vs_unif),
            "rl_choice_dist": {a: int(choice_dist[i]) for i, a in enumerate(args.agents)},
        }
        print(f"\n  --- {ds_name} (n_eval={n_eval}) ---")
        print(f"  {'method':22s} {'CRPS':>8s} {'cov80':>8s} {'n_calls':>8s}")
        for a in args.agents:
            m = per_agent[a]
            print(f"  {a:22s} {m['CRPS']:8.4f} {m['coverage_80']:8.4f} {'1':>8s}")
        print(f"  {'uniform':22s} {uniform_m['CRPS']:8.4f} {uniform_m['coverage_80']:8.4f} {len(args.agents):>8.0f}")
        print(f"  {'rl_orchestrator':22s} {rl_metrics['CRPS']:8.4f} {rl_metrics['coverage_80']:8.4f} {result['mean_n_calls']:>8.2f}")
        print(f"  -> RL vs best({best_single}): {rl_delta:+.2f}%")
        print(f"  -> RL vs uniform: {rl_vs_unif:+.2f}%")
        print(f"  -> RL picks: {per_dataset[ds_name]['rl_choice_dist']}")

    print("\n=== Path C verdict ===")
    mean_delta = float(np.mean(rl_deltas))
    print(f"Mean RL CRPS delta vs best-single across datasets: {mean_delta:+.2f}%")
    print()
    print("Interpretation:")
    print(f"  >= +3% mean    -> method works under no-shift; val/test shift was the bottleneck. Pursue Path B (top-tier push).")
    print(f"  -2% to +3%     -> method extracts modest gain even under no-shift; ceiling is the method itself.")
    print(f"                    Path A (workshop) is the realistic call; calibration story still holds.")
    print(f"  < -2%          -> method actively hurts even with matched training. Pivot needed (Path D).")
    print()
    actual = "B" if mean_delta >= 3 else ("A" if mean_delta >= -2 else "D")
    print(f"Recommended path given this verdict: {actual}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {"config": vars(args), "training_history": history, "per_dataset": per_dataset,
             "mean_rl_delta_vs_best_single_pct": mean_delta, "recommended_path": actual},
            f, indent=2, default=float,
        )
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    sys.exit(main())
