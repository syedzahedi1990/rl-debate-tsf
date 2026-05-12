# Colab pilot — cells to paste

Assumes the repo is hosted at a GitHub URL. Replace `YOUR_REPO_URL` below.

## Cell 1 — environment

```bash
!nvidia-smi
!python --version
```

## Cell 2 — clone

```bash
%cd /content
!rm -rf rl-debate-tsf
!git clone YOUR_REPO_URL rl-debate-tsf
%cd /content/rl-debate-tsf
```

## Cell 3 — install (≈ 2 min)

```bash
# Colab Pro ships PyTorch, numpy, pandas — skip those.
!pip -q install statsmodels properscoring chronos-forecasting transformers accelerate hydra-core omegaconf einops gymnasium
```

## Cell 4 — unit tests

```bash
!pytest -q tests/
```

Expected: 5 passed.

## Cell 5 — download ETTh1

```bash
!python scripts/download_etth1.py
```

## Cell 6 — smoke test

```bash
!python scripts/smoke_etth1.py --windows 32 --horizon 96
```

Expected output: prints MSE / MAE / CRPS / coverage_80, and `[ok] smoke test passed.` at the end. CRPS for ARIMA on ETTh1 H=96 should be in the ballpark of 0.4–0.8 on the z-scored scale; coverage_80 should be roughly 0.5–0.9.

If this works → we proceed to add the Chronos-Bolt agent and the naive-ensemble Gate-1 check in the next iteration.
