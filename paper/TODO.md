# Paper writeup TODOs

Working draft: `main.tex` (compile-once skeleton).

## Path A (workshop) — required for submission

- [ ] **Pull canonical numbers** from latest `results/pilot_gate2.json`,
  `results/diag_within_dataset.json`, `results/diag_conformal.json` into
  Table 1 (per-agent + oracle) and Table 2 (routing vs conformal vs
  ensembles).
- [ ] **Figure 1**: schematic of Sequential Refinement (manually drawn or tikz).
- [ ] **Figure 2**: bar chart per-agent CRPS / cov80 across 4 ETT datasets.
- [ ] **Figure 3**: oracle headroom + RL agent-pick distribution per dataset.
- [ ] **Figure 4**: training curve (mean_return, mean_ep_len) over PPO iters.
- [ ] **Related Work section** body text (intro currently stubbed). Cite at minimum:
  Du et al. (multi-agent debate), Puppeteer, Router-R1, Tan et al.,
  PatchTST, iTransformer, TimeMixer, Chronos / Moirai / TimesFM,
  Vovk / Romano (conformal), Should-we-be-going-MAD.
- [ ] **Appendix A**: with/without dataset_id ablation (sensitivity).
- [ ] **Appendix B**: PPO hyperparameters, panel agent configs, eval protocol.
- [ ] Decide venue: ICLR Workshop (March deadline) vs NeurIPS Time Series
  Workshop (around Oct) vs TMLR (rolling). Page limit: 4 (ICLR ws) /
  8 (TMLR initial).

## Path B (top-tier-receptive expansion) — required for AAAI/AISTATS/NeurIPS D&B

- [ ] **Dataset expansion**: Weather, Electricity (ECL), Traffic. Optional: M4
  Monthly, GIFT-Eval subset.
- [ ] **Conv1D state encoder**: raw lookback through a small Conv1D to extract
  conditional regime features. Compare to hand-crafted features.
- [ ] **Multiple seeds** (5) + Wilcoxon signed-rank or paired t-tests.
- [ ] **Conformal baselines fully integrated** in headline tables (have the
  code via `diag_conformal.py`; need to fold into the comparison table).
- [ ] **Long-horizon variants**: re-run at H=192, 336, 720.
- [ ] **Multivariate target**: extend the panel and metrics to features='M'.
- [ ] **Fix Qwen-LLMTime** (cov80 0.25 due to sample-collapse; needs
  temperature/parsing fix) and add to panel as a 5th agent.
- [ ] **Theoretical sketch**: under what assumptions does the learned router
  beat best-single? Even a weak guarantee strengthens the paper.

## Decision points

After Path B's first results land:

- If results+expansion clearly clear a top-tier-receptive bar
  (e.g., consistent improvement on 5+ datasets, multiple seeds significant),
  target AAAI / AISTATS / TMLR / NeurIPS D&B.
- If results remain modest (current ETT pattern repeated), submit workshop
  draft from Path A and ship a stronger version to a future round.

## Status snapshot (regenerate as numbers change)

| Result | Status | Source |
|---|---|---|
| Per-agent + oracle on ETT | Available | `results/diag_oracle.json` |
| Gate 2 RL eval on ETT | Available | `results/pilot_gate2.json` |
| Path C same-distribution | Available | `results/diag_within_dataset.json` |
| Conformal baseline | Pending (run after this push) | `results/diag_conformal.json` |
| Multi-seed verification | Pending | n/a |
| Conv1D feature encoder | Pending | n/a |
| Extended datasets | Pending | n/a |
