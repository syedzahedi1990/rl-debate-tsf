# rl-debate-tsf

Distributional Debate: RL Orchestration over Predictive Distributions for Time Series Forecasting.

See `DESIGN.md` for the locked design, MDP, baselines, and pilot gates.

## Layout

- `distdeb/` — package: agents, orchestrator, env, aggregation, data, eval
- `baselines/` — adapters for MAD, DyLAN, majority-vote
- `third_party/Time-Series-Library/` — vendored TSLib (data loaders + DL baselines)
- `scripts/` — entry points (smoke tests, data download, pilot runs)
- `configs/` — Hydra configs per experiment
- `tests/` — pytest

## Quick start (local, no GPU — premise sanity check)

```
pip install -r requirements.txt
python scripts/download_etth1.py
python scripts/smoke_etth1.py
```

This runs the ARIMA agent on ETTh1 H=96 and reports CRPS / MAE — sanity check that the data loader and eval harness work end-to-end before we add LLM agents.

## Colab (A100)

See `scripts/colab_pilot.ipynb` (added after the local smoke passes).
