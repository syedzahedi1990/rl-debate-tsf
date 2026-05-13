# Calibration-Aware Adaptive Aggregation for Foundation Time-Series Forecasters

**Status:** Design v2 locked 2026-05-13. Pivoted from CRPS-headline to calibration-headline after Gate 1a–e replicated the finding that no static mixture beats best single Chronos on CRPS at H=96 on ETT-small, but every ensemble consistently restores near-nominal coverage. Oracle headroom analysis (5.94% mean, 7.62% excl. ETTm2) supports a cautious build of the orchestrator.

## 1. Contribution claim

Foundation TSF models (Chronos, Moirai, TimesFM) achieve strong point accuracy but are **systematically miscalibrated** — Chronos cov80 ≈ 0.69, Moirai ≈ 0.78 on a 0.80 nominal target across ETT-small. Per-window oracle analysis shows the best agent varies across windows on 3/4 datasets, with 6–9% CRPS headroom over best-single available to a perfect router.

We propose an RL-orchestrated **sequential refinement protocol**: starting from a cheap forecast, the policy decides whether to halt or invoke an additional agent at each step, accumulating quantile forecasts in a running ensemble. The policy is trained to minimize CRPS at low compute cost; **near-nominal coverage emerges as a property of the diverse panel ensemble**, and we additionally report calibration-conditioned variants of the reward.

**The headline empirical claim is calibration**: matching the most accurate single-agent's CRPS *while* achieving nominal coverage *and* using fewer agent calls than the full panel ensemble.

**The MDP-level differentiation from Puppeteer (Wei et al., 2025) and Router-R1 (Tang et al., NeurIPS'25)** is preserved:
- They output text tokens; we output predictive distributions over a horizon.
- Their state has no notion of calibration or interval width; ours does.
- They terminate on `<answer>` / Terminator agent; we terminate on a learned HALT that conditions on ensemble width.

## 2. Differentiation from prior work

| Axis | Puppeteer (2505.19591) | Router-R1 (2506.09033) | **DistDeb (ours)** |
|---|---|---|---|
| Output | Text (code, reasoning) | Text (QA answer, EM-scored) | Predictive distribution over R^H |
| Aggregation | Majority vote on text outputs | Router LLM rewrites with stacked context | Learned distributional combinator (quantile stacking + conformal calibration) |
| State features | "Aggregated text context" (vague) | Query + text history | Text history + **regime embedding** + ensemble predictive moments + **calibration stats** |
| Halt criterion | Terminator agent invoked / FLOP budget | Router emits `<answer>` token | **Predictive-interval width below learned threshold AND coverage matches nominal** |
| Action space | Pick agent (model, reasoning pattern, tool) | `<think>` / `<search> LLM_i` / `<answer>` | Pick agent + distributional combinator weights + halt |
| Reward | `r − λ·log(1+t/φ)·FLOPs`, REINFORCE no baseline | `R_format + (1−α)·EM + α·R_cost`, PPO | `−CRPS − λ_calls·n_calls − λ_cal·CalibrationGap`, PPO |
| RL algo | REINFORCE (no KL, no baseline) | PPO (veRL) | PPO with KL anchor to fixed-schedule policy |
| Inter-agent interaction | Independent agents, vote | Sequential stacking | **Cross-agent rebuttal** (agents see prior forecasts + critique structure) |
| Domain | Reasoning/QA | QA | Time series forecasting |

Three things prior work **cannot** do, by construction of their MDPs:

- Express uncertainty over a forecast horizon (their outputs are tokens).
- Use calibration of a predictive distribution as a halt signal.
- Route based on regime structure of an input signal.

## 3. MDP formulation (precise — v2 sequential refinement)

**State** s_t at refinement step t:
- `window_features`: lookback summary statistics (mean, std, min, max, range, skew, autocorr lag-1, stationarity stat). Per-window normalized.
- `ensemble_quantiles`: flattened current ensemble's quantile forecasts (Q × H), zero if no agents called yet.
- `ensemble_width`: scalar — mean width of the 80% interval across horizon.
- `called_mask`: binary (N_agents,) — which agents have been invoked.
- `n_calls`: scalar.

**Action** a_t ∈ {0, 1, ..., N_agents}:
- `0` = HALT — terminate and emit current ensemble as final forecast.
- `1..N_agents` = call agent i (invalid if already called → masked or penalized).

**Transition**: deterministic. If action = call i, agent i's *cached* quantile forecast for the current window is added to ensemble (equal-weight aggregation across called agents). If action = HALT or all agents called, terminal.

**Reward**:
- Per-step: −λ_cost · cost(agent_i) when calling agent i. Encourages early halt.
- Terminal: −CRPS(F_final, y) for window y. Implicitly rewards calibrated forecasts because CRPS penalizes both under- and over-coverage.
- Optional calibration term: −λ_cal · |emp_cov_80(batch) − 0.80| computed batch-level during training.

**Horizon (RL)**: at most N_agents steps (one per agent), naturally bounded.

**Policy architecture**: 2-layer MLP, 128 hidden, ReLU. Inputs: flattened state (~50 dims). Outputs: softmax over N_agents+1 actions + scalar value head. ~30k parameters. Trains in minutes on A100.

**RL algorithm**: PPO with clipped surrogate objective. Single policy trained on val windows mixed across all 4 datasets (≥ 1000 windows total). Evaluated on test windows per-dataset. KL anchor to a uniform-random policy for the first 100 updates to prevent collapse.

**Action masking**: invalid actions (already-called agents) are masked from the policy distribution at inference and training. The HALT action is always available.

## 4. Agent panel

Heterogeneous on purpose:

1. **Trend specialist** — Chronos-Bolt-base, zero-shot, prompted with detrended series.
2. **Seasonality specialist** — Chronos-Bolt-base, zero-shot, prompted with deseasonalized residual.
3. **Anomaly/regime specialist** — Qwen2.5-7B with anomaly-detection prompt, outputs forecast conditioned on detected regime.
4. **Generalist LLM forecaster** — Time-LLM checkpoint or Qwen2.5-7B in LLMTime style.
5. **Statistical anchor** — frozen ARIMA/ETS via `statsmodels`. Cheap and grounding; included to falsify "Are LMs Actually Useful for TSF?" (Tan et al. 2024) — if the orchestrator's optimal policy is "just call the statistical anchor every time", we report that honestly.
6. **Optional frontier expert** — Claude/GPT API call, gated by RL on cost. Demonstrates the cost-routing axis.

All agents output quantile forecasts {0.1, 0.5, 0.9} over the horizon. We standardize the output format to make distributional aggregation tractable.

## 5. Baselines

**Specialist TSF (no debate):**
- DLinear, PatchTST, iTransformer, TimeMixer
- GPT4TS, Time-LLM, Chronos-Bolt-base

**Debate / orchestration (the headline comparison):**
- Single-agent CoT (one LLM forecaster, one call)
- Self-consistency / **cost-matched majority vote** (the dangerous baseline — same n_calls as DistDeb, averaged)
- Du et al. MAD adapted to TSF (fixed 3 rounds, each agent revises seeing peers' forecasts)
- DyLAN-style heuristic dynamic orchestrator
- **Random orchestrator** (ablation: same panel, random schedule, same budget)
- **Fixed-schedule orchestrator** (ablation: round-robin over our panel — isolates the contribution of *learning* the schedule)
- **Greedy uncertainty halt** (ablation: stop when interval width below fixed threshold, no RL)

## 6. Datasets and metrics

**Pilot (this week):**
- ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity
- L = 96; H ∈ {96, 192, 336, 720}
- 3 seeds
- Metrics: MSE, MAE (z-scored), CRPS, n_calls per forecast, wall-clock

**Full paper:**
- Add Traffic, Exchange, ILI (full Informer suite)
- Add GIFT-Eval zero-shot subset (foundation-model bar)
- Add M4 Monthly + Hourly (LLM-TSF convention)
- 5 seeds
- Statistical tests: Wilcoxon signed-rank across (dataset, horizon) pairs; critical-difference diagram

**Eval harness:** Fork `thuml/Time-Series-Library`. **Fix the `drop_last=True` test-loader bug** (Qiu et al. 2024) and call this out in §5 — pre-empts a reviewer complaint.

## 7. Critiques to address head-on (must be in the paper)

- **"Are LMs Actually Useful for TSF?" (Tan et al., NeurIPS'24).** Include a panel where the LLM-based agents are replaced by frozen statistical forecasters and show the orchestrator + LLM panel still wins. If it doesn't, report it.
- **"Should we be going MAD?" (Smit et al., ICML'24) and "Debate or Vote" (2508.17536).** Cost-matched majority vote is the headline baseline. If we don't beat it on CRPS at matched n_calls, the paper is dead.
- **MAD over-flips correct answers (ICLR 2025 blog).** Report answer-stability metric: fraction of forecasts that change sign / direction across debate rounds.

## 8. Pilot gates — final state (locked 2026-05-13)

- **Gate 1**: FAILED. No static mixture beats best single Chronos on ETT-small at H=96.
- **Gate 1b (oracle headroom)**: +5.94% mean (+7.62% excl. ETTm2). Decorrelation exists but is window-stochastic.
- **Gate 2 (standard val/test split)**: FAILED on CRPS criterion. PASSED on calibration (cov80 within ±0.03 of 0.80 on 3/4 datasets) and HALT-exercising. Mean CRPS delta -17.5% vs best-single.
- **Path C (same-distribution diagnostic)**: -1.36% mean CRPS delta. The method has a ceiling — even matched-distribution training cannot extract the oracle headroom.
- **Conv1D ablation**: tested in place of hand-crafted features; *worsened* standard-split CRPS to -34.1% mean. Conv1D overfits val; raw-lookback features don't help under distribution shift.
- **Conformal baseline (the genuinely interesting finding)**: split-conformal *overcorrects* on 3/4 ETT datasets (cov80 0.95+), inflating intervals and harming CRPS. The RL orchestrator does not overcorrect (mean cov80 deviation 0.022 vs 0.093 for conformal best-single).

**Verdict**: workshop-tier paper. Headline = calibration-without-conformal-overcorrection, supporting claim = matched-uniform-CRPS at 30% lower compute. Ceiling on CRPS over best-single documented as a contribution rather than hidden as a limitation.

**Next-step priorities (in order)**:
1. Finalize workshop draft using current results — target ICLR Workshop or TMLR.
2. Expand to Weather / Electricity / Traffic to broaden the empirical claim before a stronger venue.
3. Multi-seed evaluation + Wilcoxon for statistical rigor.

No further method iteration on ETT-small at H=96.

## 9. Risks (known)

- **Chronos-Bolt as panel members may already be near-SOTA on these benchmarks.** Adding LLM debate to a strong specialist may not help. If gate 1 fails because Chronos alone is too strong, reframe to weaker-panel composition or harder benchmarks (regime-shift datasets).
- **MPDF (2509.03817) is the closest concurrent work.** RL-trained debate on reasoning, no TS, decentralized per-agent meta-actions. Position as concurrent prior art, emphasize centralized orchestrator + distributional output.
- **GPU bottleneck for RL training.** Cache agent outputs deterministically (seeded sampling) so PPO rollouts replay cached forecasts; RL training pays no LLM cost after the first epoch over the training set. This is the key cost-control trick.

## 10. Repository layout (planned)

```
rl-debate-tsf/
├── DESIGN.md              # this file
├── README.md
├── third_party/
│   └── Time-Series-Library/   # fork, drop_last fix applied
├── distdeb/
│   ├── agents/            # panel members
│   ├── orchestrator/      # policy net, PPO trainer
│   ├── env/               # RL env wrapping the forecast task
│   ├── aggregation/       # quantile stacking, conformal
│   ├── data/              # dataset loaders (delegate to TSLib)
│   └── eval/              # CRPS, n_calls, calibration metrics
├── baselines/             # MAD, DyLAN, majority-vote adapters
├── configs/               # hydra configs per experiment
├── scripts/               # train_pilot.sh, eval.sh
└── results/               # logged outputs
```
