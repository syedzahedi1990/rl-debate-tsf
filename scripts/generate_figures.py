"""Generate paper figures from results JSONs.

Saves PDFs to paper/figures/. Run after gate / diag scripts have populated
results/. Idempotent.

Usage:
  python scripts/generate_figures.py
"""

from __future__ import annotations

import json
import os
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = "paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def _safe_load(path):
    if not os.path.exists(path):
        print(f"  [miss] {path}")
        return None
    with open(path) as f:
        return json.load(f)


def fig_method_comparison(conformal_path="results/diag_conformal.json"):
    """Per-dataset CRPS + cov80 bars: naive vs conformal vs uniform vs RL."""
    data = _safe_load(conformal_path)
    if data is None:
        return
    summary = data["summary"]
    datasets = list(summary.keys())

    method_labels = [
        ("best naive single",  "naive_best"),
        ("best conformal single", "conf_best"),
        ("uniform (naive)",    "unif_naive"),
        ("uniform (conformal)", "unif_conf"),
        ("RL orchestrator",    "rl"),
    ]
    methods = [m[1] for m in method_labels]
    labels = [m[0] for m in method_labels]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(methods)))

    def get_value(d, m, key):
        if m == "naive_best":
            return d["naive"][d["best_naive_agent"]][key]
        if m == "conf_best":
            return d["conformal"][d["best_cal_agent"]][key]
        if m == "unif_naive":
            return d["uniform_naive"][key]
        if m == "unif_conf":
            return d["uniform_conformal"][key]
        if m == "rl":
            return d["rl"][key] if d["rl"] else np.nan
        raise ValueError(m)

    fig, (axc, axv) = plt.subplots(1, 2, figsize=(11, 3.6))
    width = 0.16
    x = np.arange(len(datasets))

    for i, (label, m) in enumerate(method_labels):
        crps = [get_value(summary[d], m, "CRPS") for d in datasets]
        cov = [get_value(summary[d], m, "coverage_80") for d in datasets]
        axc.bar(x + i * width, crps, width, color=colors[i], label=label)
        axv.bar(x + i * width, cov, width, color=colors[i], label=label)
    for ax in (axc, axv):
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(datasets)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axc.set_ylabel("CRPS (lower is better)")
    axc.set_title("(a) CRPS")
    axv.set_ylabel("empirical 80% coverage")
    axv.set_title("(b) coverage (nominal 0.80)")
    axv.axhline(0.80, color="black", linewidth=0.7, linestyle="--", label="nominal")
    axv.legend(loc="upper left", fontsize=7, ncol=1, frameon=False)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "method_comparison.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [ok] {out}")


def fig_oracle_headroom(diag_path="results/diag_oracle.json"):
    data = _safe_load(diag_path)
    if data is None:
        return
    datasets = [r["dataset"] for r in data]
    best = [r["best_single_crps"] for r in data]
    oracle = [r["oracle_crps"] for r in data]
    headroom = [r["oracle_headroom_pct"] for r in data]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    width = 0.35
    x = np.arange(len(datasets))
    ax.bar(x - width / 2, best, width, color="#4c72b0", label="best single")
    ax.bar(x + width / 2, oracle, width, color="#dd8452", label="per-window oracle")
    for i, h in enumerate(headroom):
        ax.text(x[i] + width / 2, oracle[i], f"+{h:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("CRPS")
    ax.set_title("Per-window oracle headroom over best single agent")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "oracle_headroom.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [ok] {out}")


def fig_training_curve(gate2_path="results/pilot_gate2.json"):
    data = _safe_load(gate2_path)
    if data is None:
        return
    history = data.get("training_history", [])
    if not history:
        return
    iters = [h["iter"] for h in history]
    ret = [h["mean_return"] for h in history]
    ep_len = [h["mean_ep_len"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.plot(iters, ret, color="#4c72b0")
    ax1.set_xlabel("PPO iteration")
    ax1.set_ylabel("mean episode return")
    ax1.set_title("(a) training return")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.plot(iters, ep_len, color="#dd8452")
    ax2.axhline(4, color="black", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("PPO iteration")
    ax2.set_ylabel("mean episode length")
    ax2.set_title("(b) calls per episode")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "training_curve.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [ok] {out}")


def fig_rl_choice_distribution(gate2_path="results/pilot_gate2.json"):
    data = _safe_load(gate2_path)
    if data is None:
        return
    per_ds = data.get("per_dataset", {})
    if not per_ds:
        return
    datasets = list(per_ds.keys())
    agents = list(next(iter(per_ds.values()))["rl_choice_dist"].keys())

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    width = 0.8 / len(agents)
    x = np.arange(len(datasets))
    colors = plt.cm.Set2(np.linspace(0, 1, len(agents)))
    for i, ag in enumerate(agents):
        counts = [per_ds[d]["rl_choice_dist"][ag] for d in datasets]
        ax.bar(x + i * width - 0.4 + width / 2, counts, width, color=colors[i], label=ag)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("# windows agent was called on")
    ax.set_title("RL orchestrator agent picks (per dataset)")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "rl_choice_distribution.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [ok] {out}")


def main():
    print("Generating paper figures...")
    fig_method_comparison()
    fig_oracle_headroom()
    fig_training_curve()
    fig_rl_choice_distribution()
    print(f"Done. PDFs in {OUT_DIR}/.")


if __name__ == "__main__":
    main()
