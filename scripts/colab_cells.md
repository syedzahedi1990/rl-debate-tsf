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
