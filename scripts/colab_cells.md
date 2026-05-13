# Colab pilot — cells to paste

Repo: https://github.com/syedzahedi1990/rl-debate-tsf

## Cell 1 — environment

```bash
!nvidia-smi
!python --version
```

## Cell 2 — clone (first time only)

```bash
%cd /content
!rm -rf rl-debate-tsf
!git clone https://github.com/syedzahedi1990/rl-debate-tsf.git
%cd /content/rl-debate-tsf
```

Subsequent iterations: just `!git pull`.

## Cell 3 — install (≈ 2 min)

```bash
!pip -q install statsmodels properscoring chronos-forecasting transformers accelerate hydra-core omegaconf einops gymnasium
```

## Cell 3b — install heterogeneous-panel deps

**If you just ran `pip install uni2ts` on a previous attempt the runtime is
in a broken state** (uni2ts downgrades torch to 2.4 while leaving Colab's
torchvision 0.25 that was built for torch 2.10). To recover:

1. Runtime → Disconnect and delete runtime
2. Runtime → Connect → fresh runtime
3. Re-run cells 1, 2, 3, 5 (clone + base deps + data download).

Then for the heterogeneous panel:

```bash
# Moirai: install uni2ts + a torchvision that matches the torch it downgrades to
!pip -q install uni2ts torchvision==0.19.1
# Qwen-LLMTime: nothing extra — transformers + accelerate (from Cell 3) are enough.
```

**Note on TimesFM**: incompatible with Python 3.12 (Colab's runtime) — 1.0.0
needs paxml (no 3.12 wheels), 1.2+ pins Python <3.12. Deferred to the final
paper run on a Py3.11 vast.ai instance.

## Cell 4 — unit tests

```bash
!pytest -q tests/
```

Expected: 5 passed.

## Cell 5 — download data

```bash
!python scripts/download_etth1.py
```

## Cell 6 — smoke test (ARIMA only, ~5 sec)

```bash
!python scripts/smoke_etth1.py --windows 32 --horizon 96
```

## Cell 6b — smoke test individual new agents

```bash
# Each takes <1 min once weights are downloaded.
!python scripts/smoke_agent.py --agent chronos_base --windows 32 --horizon 96
!python scripts/smoke_agent.py --agent moirai --windows 32 --horizon 96
```

## Persist forecast cache to Google Drive (recommended)

Colab's `/content` is wiped on runtime disconnect — every pilot re-run loses
the cached forecasts. To avoid this, mount Drive and set the cache root:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ['DISTDEB_CACHE_ROOT'] = '/content/drive/MyDrive/rl-debate-tsf-cache'
!mkdir -p $DISTDEB_CACHE_ROOT
```

Then every subsequent `python scripts/...` call (in the same notebook) will
read/write the persistent cache. To override per-invocation: `--cache-root /content/drive/MyDrive/rl-debate-tsf-cache`.

## Gate 2 — train + eval the RL orchestrator

After pilot_gate1e has populated the cache for the 4-agent panel:

```bash
!python scripts/pilot_gate2.py --n-iters 200 --windows 256
```

Time: a few minutes (rollouts are free — all forecasts are cached). Reports
per-dataset CRPS / coverage_80 / mean_n_calls for the trained RL policy
vs. each single agent and the uniform ensemble.

Gate 2 PASS criteria (per DESIGN.md S8):
  - cov80 within +/-0.03 of 0.80 on >= 3/4 datasets
  - CRPS within 2% of best single on all datasets
  - mean n_calls < N_agents (HALT is being used)

## Cell 7 — Gate 1: smaller subset (fast sanity, ~1-2 min)

```bash
!python scripts/pilot_gate1.py --windows 256 --horizon 96
```

This runs ARIMA + Chronos-Bolt-base on 256 ETTh1 test windows and reports
single-agent metrics + equal-weight ensemble. Gate 1 PASS = ensemble beats
the best single agent on CRPS.

## Cell 8 — Gate 1: full test set (~10 min — ARIMA is the bottleneck)

```bash
!python scripts/pilot_gate1.py --windows -1 --horizon 96
```

Use this once Cell 7 looks reasonable. ARIMA takes ~10 min on ~5800 windows;
Chronos-Bolt should be seconds on an A100.

If ARIMA is too slow and you just want Chronos numbers:

```bash
!python scripts/pilot_gate1.py --windows -1 --horizon 96 --skip-arima
```

## What I'm watching for

- Both agents return finite CRPS, coverage_80 ∈ [0, 1].
- Chronos-Bolt CRPS should be lower than ARIMA's (it's a much stronger model).
- The verdict line will say `GATE 1 PASS` or `GATE 1 FAIL`. Paste the full output
  back to me — the numbers tell us where to push next.
