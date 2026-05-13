# Paper writeup TODOs

Working draft: `main.tex` (compile-once skeleton).

## Path A (workshop) — submission checklist

- [x] **Pull canonical numbers** from latest results JSONs into the
  two main tables.
- [x] **Related Work section** body text.
- [x] **Method section** body text.
- [x] **Discussion + Conclusion** sections.
- [x] **Appendix A** (dataset-id ablation), **B** (hyperparameters),
  **D** (RL choice distribution), **E** (shuffled-split conformal sanity),
  **F** (Conv1D ablation).
- [x] **References** complete in `references.bib`.
- [ ] **Run `scripts/generate_figures.py`** to produce PDFs, then
  uncomment `\includegraphics` blocks in `main.tex`.
- [ ] **Appendix C**: sensitivity to test window count (currently TODO marker).
- [ ] **Method schematic** (Figure 1): sequential-refinement diagram. Hand-drawn
  or `tikz`. Not blocking submission.
- [ ] **Final proofread** for length: ICLR/NeurIPS workshop limits ~8pp,
  current draft is ~6pp. TMLR allows up to 12pp.
- [ ] **Decide venue and apply stylesheet**:
  - **TMLR (recommended, rolling)**: use `tmlr.sty`; current article-class
    draft converts cleanly.
  - **ICLR Workshop**: download next-round style file when CFP opens.
  - **NeurIPS Workshop**: same.

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
