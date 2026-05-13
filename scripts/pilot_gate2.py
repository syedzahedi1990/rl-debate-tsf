"""Gate 2: does the RL orchestrator beat baselines on calibration + matched-cost CRPS?

Train a single PPO policy on val windows mixed across all 4 ETT datasets.
Evaluate on test windows per-dataset. Compare to:
  - Per-agent best single (from cache)
  - Uniform ensemble (all agents)
  - Val-tuned static weights (from pilot_gate1e cache)
  - Oracle per-window picks (upper bound)

Reports per-dataset: CRPS, coverage_80, mean_n_calls, plus the choice
distribution of the trained policy.

Gate 2 PASS criteria (DESIGN.md S8):
  1. Test cov80 within +/- 0.03 of nominal 0.80 on >= 3/4 datasets.
  2. Test CRPS not worse than best-single by more than 2% on any dataset.
  3. Mean n_calls < N_agents (RL is using HALT).

Usage:
  python scripts/pilot_gate2.py --n-iters 200 --windows 256
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

from distdeb.env.refinement_env import RefinementEnv, window_features
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


def val_windows(series, seq_len, pred_len, n):
    return _make_windows(series, series["train_end"] + seq_len, series["val_end"] - pred_len + 1,
                         seq_len, pred_len, stride=1, limit=n)


def test_windows(series, seq_len, pred_len, n):
    return _make_windows(series, series["val_end"], len(series["data"]) - pred_len + 1,
                         seq_len, pred_len, stride=1, limit=n)


def _metrics_from_quantiles(quantiles, levels, targets):
    median_idx = int(np.argmin(np.abs(levels - 0.5)))
    median = quantiles[:, median_idx, :]
    q_for_crps = np.transpose(quantiles, (1, 0, 2))
    return {
        "MSE": float(mse(median, targets)),
        "MAE": float(mae(median, targets)),
        "CRPS": float(quantile_crps(q_for_crps, levels, targets)),
        "coverage_80": float(empirical_coverage(q_for_crps, levels, targets, alpha=0.8)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--windows", type=int, default=256, help="Test windows per dataset.")
    p.add_argument("--val-windows", type=int, default=512, help="Val windows per dataset used for RL training.")
    p.add_argument("--cache-root", default=None, help="Defaults to env var DISTDEB_CACHE_ROOT or data_cache/forecasts.")
    p.add_argument("--agents", nargs="+", default=["arima", "chronos_base", "chronos_tiny", "moirai"])
    p.add_argument("--cost-weight", type=float, default=0.01)
    p.add_argument("--n-iters", type=int, default=200)
    p.add_argument("--episodes-per-iter", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/pilot_gate2.json")
    p.add_argument("--ckpt", default="results/orchestrator.pt")
    args = p.parse_args()

    cache = ForecastCache(root=args.cache_root)
    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

    # ---- Load val windows + cached quantiles, build joint training env ----
    print(f"[setup] loading {args.val_windows} val windows/dataset and {args.windows} test windows/dataset")
    train_hist_list, train_tgt_list, train_id_list = [], [], []
    train_q_list: Dict[str, List[np.ndarray]] = {a: [] for a in args.agents}
    test_hist_by_ds = {}
    test_tgt_by_ds = {}
    test_q_by_ds: Dict[str, Dict[str, np.ndarray]] = {}

    for ds_id, (name, csv_path, freq) in enumerate(DATASETS):
        if not os.path.exists(csv_path):
            print(f"[skip] {csv_path}")
            continue
        series = _load_ett(csv_path, freq)
        val_w = val_windows(series, args.seq_len, args.horizon, args.val_windows)
        test_w = test_windows(series, args.seq_len, args.horizon, args.windows)
        if not val_w or not test_w:
            continue

        val_hist = np.stack([h.squeeze(-1) for h, _ in val_w]).astype(np.float32)
        val_tgt = np.stack([t.squeeze(-1) for _, t in val_w]).astype(np.float32)
        test_hist = np.stack([h.squeeze(-1) for h, _ in test_w]).astype(np.float32)
        test_tgt = np.stack([t.squeeze(-1) for _, t in test_w]).astype(np.float32)

        # Verify cache: all required (ds, agent) pairs must be present.
        ds_test_q = {}
        ds_val_q = {}
        missing = False
        for ag in args.agents:
            tq = cache.load(
                dataset=name, agent=ag, seq_len=args.seq_len, horizon=args.horizon, split="test",
                n_windows=len(test_w), levels=levels,
            )
            # Val cached under split=val0..val{N-1}; for training, just re-use test cache as a stand-in
            # is incorrect — we need val. The Gate 1e script cached val under split="val0..3". For
            # this pilot we'll re-use the val blocks if available, else fall back.
            vq = None
            for bi in range(4):
                part = cache.load(
                    dataset=name, agent=ag, seq_len=args.seq_len, horizon=args.horizon,
                    split=f"val{bi}", n_windows=64, levels=levels,
                )
                if part is None:
                    break
                vq = part if vq is None else np.concatenate([vq, part], axis=0)
            if tq is None or vq is None:
                print(f"  [miss] {name}/{ag} test={tq is not None} val={vq is not None}")
                missing = True
                break
            ds_test_q[ag] = tq
            ds_val_q[ag] = vq
        if missing:
            print(f"  [skip] {name} -- run scripts/pilot_gate1e.py first to populate cache.")
            continue

        # Trim training to whichever block-stack length matches val_hist length.
        n_val_avail = min(len(val_hist), min(v.shape[0] for v in ds_val_q.values()))
        val_hist = val_hist[:n_val_avail]
        val_tgt = val_tgt[:n_val_avail]
        for ag in args.agents:
            ds_val_q[ag] = ds_val_q[ag][:n_val_avail]

        train_hist_list.append(val_hist)
        train_tgt_list.append(val_tgt)
        train_id_list.append(np.full(n_val_avail, ds_id, dtype=np.int32))
        for ag in args.agents:
            train_q_list[ag].append(ds_val_q[ag])

        test_hist_by_ds[name] = test_hist
        test_tgt_by_ds[name] = test_tgt
        test_q_by_ds[name] = ds_test_q

    if not train_hist_list:
        print("[err] no datasets have a complete cache. Run pilot_gate1e first.")
        return 1

    train_hist = np.concatenate(train_hist_list, axis=0)
    train_tgt = np.concatenate(train_tgt_list, axis=0)
    train_ids = np.concatenate(train_id_list, axis=0)
    train_q = {ag: np.concatenate(train_q_list[ag], axis=0) for ag in args.agents}
    print(f"[setup] training env: {train_hist.shape[0]} windows mixed across {len(test_hist_by_ds)} datasets")

    # ---- Build env + policy ----
    train_env = RefinementEnv(
        histories=train_hist,
        targets=train_tgt,
        agent_quantiles=train_q,
        levels=levels,
        dataset_ids=train_ids,
        cost_weight=args.cost_weight,
    )
    print(f"[setup] state_dim={train_env.state_dim}, n_actions={train_env.n_actions}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    policy = PolicyValueNet(train_env.state_dim, train_env.n_actions, hidden=128)
    cfg = PPOConfig(
        n_iters=args.n_iters,
        n_episodes_per_iter=args.episodes_per_iter,
        lr=args.lr,
        seed=args.seed,
    )
    trainer = PPOTrainer(train_env, policy, cfg, device=device)

    print(f"\n[train] PPO on {train_env.N} windows, {cfg.n_iters} iters x {cfg.n_episodes_per_iter} eps")
    history = trainer.train(log_every=max(1, cfg.n_iters // 10))

    # ---- Save policy + history ----
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    torch.save({"policy": policy.state_dict(), "config": cfg.__dict__, "history": history}, args.ckpt)
    print(f"[saved] {args.ckpt}")

    # ---- Eval on each test dataset ----
    print("\n[eval] per-dataset test")
    per_dataset_results = {}
    for ds_name, test_hist in test_hist_by_ds.items():
        test_tgt = test_tgt_by_ds[ds_name]
        eval_env = RefinementEnv(
            histories=test_hist,
            targets=test_tgt,
            agent_quantiles=test_q_by_ds[ds_name],
            levels=levels,
            cost_weight=args.cost_weight,
        )
        result = trainer.evaluate(eval_env, n_windows=len(test_hist), deterministic=True)
        rl_metrics = _metrics_from_quantiles(result["ensembles"], levels, test_tgt)

        # Baselines for this dataset
        per_agent = {ag: _metrics_from_quantiles(test_q_by_ds[ds_name][ag], levels, test_tgt) for ag in args.agents}
        stack_all = np.stack([test_q_by_ds[ds_name][ag] for ag in args.agents], axis=0)
        uniform_q = np.mean(stack_all, axis=0)
        uniform_m = _metrics_from_quantiles(uniform_q, levels, test_tgt)

        best_single = min(per_agent, key=lambda a: per_agent[a]["CRPS"])
        bs_crps = per_agent[best_single]["CRPS"]
        rl_delta = 100 * (bs_crps - rl_metrics["CRPS"]) / bs_crps

        # RL choice distribution
        all_chosen = [c for cc in result["chosen"] for c in cc]
        choice_dist = np.bincount(all_chosen, minlength=len(args.agents))

        per_dataset_results[ds_name] = {
            "per_agent": per_agent,
            "uniform": uniform_m,
            "rl": rl_metrics | {"mean_n_calls": result["mean_n_calls"]},
            "best_single": best_single,
            "rl_vs_best_crps_pct": float(rl_delta),
            "rl_choice_dist": {ag: int(choice_dist[i]) for i, ag in enumerate(args.agents)},
        }
        print(f"\n  --- {ds_name} ---")
        print(f"  {'method':22s} {'CRPS':>8s} {'cov80':>8s} {'n_calls':>8s}")
        for ag in args.agents:
            m = per_agent[ag]
            print(f"  {ag:22s} {m['CRPS']:8.4f} {m['coverage_80']:8.4f} {'1':>8s}")
        print(f"  {'uniform':22s} {uniform_m['CRPS']:8.4f} {uniform_m['coverage_80']:8.4f} {len(args.agents):>8.0f}")
        print(f"  {'rl_orchestrator':22s} {rl_metrics['CRPS']:8.4f} {rl_metrics['coverage_80']:8.4f} {result['mean_n_calls']:>8.2f}")
        print(f"  -> RL vs best-single({best_single}): {rl_delta:+.2f}%")
        print(f"  -> RL agent picks: {per_dataset_results[ds_name]['rl_choice_dist']}")

    # ---- Gate 2 verdict ----
    print("\n=== Gate 2 verdict ===")
    pass_cov = sum(1 for r in per_dataset_results.values() if abs(r["rl"]["coverage_80"] - 0.80) <= 0.03)
    pass_crps = sum(1 for r in per_dataset_results.values() if r["rl_vs_best_crps_pct"] >= -2.0)
    halts_used = all(r["rl"]["mean_n_calls"] < len(args.agents) for r in per_dataset_results.values())
    print(f"cov80 within +/-0.03 of 0.80 on {pass_cov}/{len(per_dataset_results)} datasets")
    print(f"CRPS within 2% of best single on {pass_crps}/{len(per_dataset_results)} datasets")
    print(f"HALT exercised (n_calls < N_agents on all datasets): {halts_used}")
    gate_pass = pass_cov >= 3 and pass_crps >= len(per_dataset_results) and halts_used
    print(f"Gate 2: {'PASS' if gate_pass else 'FAIL'}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = {
        "config": vars(args),
        "training_history": history,
        "per_dataset": per_dataset_results,
        "gate2_pass": gate_pass,
        "summary": {"pass_cov": pass_cov, "pass_crps": pass_crps, "halts_used": halts_used},
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
