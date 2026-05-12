# Distributional Debate: RL Orchestration over Predictive Distributions for Time Series Forecasting

**Status:** Design locked 2026-05-12. Pilot in progress.

## 1. Contribution claim

We propose **Distributional Debate (DistDeb)**: an RL-trained orchestrator over a heterogeneous panel of time-series forecasters that

1. routes on a **regime-aware state** (state includes a learned regime embedding from the series, recent volatility, autocorrelation, stationarity statistics — not just text history),
2. halts on a **calibration-aware criterion** (stop when predictive-interval width drops below a learned threshold *and* empirical coverage matches nominal),
3. aggregates via **learned distributional combinators** (action space includes quantile-stacking weights and conformal-calibration parameters, not just "which agent to call next"),
4. trains end-to-end with a **CRPS-grounded cost-aware reward**.

The orchestrator's MDP outputs a *predictive distribution over a horizon*, not a text token. This makes DistDeb a fundamentally different MDP from Puppeteer (Wei et al., 2025) and Router-R1 (Tang et al., NeurIPS'25), which both produce discrete text outputs and use text-only state.

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

## 3. MDP formulation (precise)

**State** s_t at debate step t:
- `series_embedding`: pretrained TS encoder output (e.g., Chronos-Bolt-tiny encoder, frozen) over the lookback window.
- `regime_features`: ADF statistic, trend slope, dominant period from FFT, recent vol, lag-1 autocorr.
- `ensemble_state`: current quantile forecasts {q_0.1, q_0.5, q_0.9} from agents called so far (concatenated), agent call mask, per-agent computational cost incurred.
- `calibration_state`: rolling empirical coverage of 80% intervals on a held-out calibration buffer (refreshed during training).
- `budget_remaining`: scalar.

**Action** a_t (factored):
- `agent_choice ∈ {0, ..., N_agents}` where `0 = HALT-and-aggregate`.
- `combiner_update ∈ R^N_agents` (simplex, softmax-parameterized): how to weight quantile forecasts in current aggregation.
- (optional) `conformal_delta ∈ R`: post-hoc shift applied to quantile width.

**Transition**: deterministic given agent output. If agent_choice = i ≠ 0, agent i is invoked, its forecast appended to ensemble_state. If agent_choice = 0, episode terminates and final aggregated distribution is produced from combiner_update.

**Reward**: episodic. At termination,
```
R = −CRPS(F_aggregate, y) − λ_calls · n_calls − λ_cal · |coverage_80 − 0.8|
```
where `F_aggregate` is the final predictive distribution and `y` is the ground-truth horizon. λ_calls and λ_cal are hyperparameters swept in ablation. Per-step shaping: −λ_calls per call to encourage early halt.

**Horizon (RL)**: max 6 debate steps (matches Puppeteer's 4-step budget — slightly higher for fair benchmarking).

**Policy architecture**: 2-layer transformer over a flattened state vector (~512-d), softmax head over agent_choice, separate softmax head over combiner weights, MLP head over conformal_delta. ~5M params. Small enough to train on a single A100 in hours; large enough to have capacity over the state we provide.

**RL algorithm**: PPO with KL anchor to a *fixed round-robin policy* (a uniform schedule over agents). The KL anchor prevents collapse to a single-agent policy during early training (the failure mode REINFORCE-no-baseline tends to hit). Borrowed from Router-R1's PPO recipe.

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

## 8. Pilot success criteria (kill / pivot / proceed)

Run pilot on ETTh1 only, H=96, 3 seeds, single A100.

- **Premise check (gate 1):** Does *any* multi-agent debate configuration on our panel beat the best single agent in CRPS at matched cost? If no → premise dead → pivot to a paper about *why* debate doesn't help TSF (still publishable, smaller venue).
- **Method check (gate 2):** Does DistDeb beat the random + fixed-schedule + greedy-halt ablations by ≥3% on CRPS at matched n_calls? If no → RL is not contributing → investigate state/reward design, retry once, then pivot.
- **Generalization check (gate 3):** Do gains hold on ETTh2 and ETTm1 with the same trained orchestrator? If no → overfitting → broaden training set, retry.

Each gate is binary. Each gate failure triggers a documented decision: retry once with a specified fix, or pivot. Total pilot budget: ≤5 days of A100 time + ≤$50 of frontier API for failure-mode analysis.

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
