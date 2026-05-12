"""On-disk forecast cache.

Agent forecasts are deterministic given (dataset, agent_name, seq_len, horizon,
split, quantile levels) plus the agent's own configuration. LLM agents in
particular are slow to run; caching their outputs once means RL training
rollouts, ablations, and re-runs are I/O bound rather than LLM-bound.

Cache layout:
  {root}/{dataset}_{agent}_L{seq_len}_H{horizon}_{split}.npz
    quantiles: (N, Q, H) float32 — N windows, Q quantile levels, H horizon
    levels:    (Q,) float32 — quantile levels (sorted ascending)

Conventions:
  - `agent` may include hyperparameter suffixes (e.g. 'qwen2.5_7b_n10_t0.7')
    so different agent configs don't collide.
  - If a cache file exists but with too few windows or different levels,
    `load` returns None and the caller recomputes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np


class ForecastCache:
    def __init__(self, root: str | Path = "data_cache/forecasts"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, dataset: str, agent: str, seq_len: int, horizon: int, split: str) -> Path:
        return self.root / f"{dataset}_{agent}_L{seq_len}_H{horizon}_{split}.npz"

    def load(
        self,
        *,
        dataset: str,
        agent: str,
        seq_len: int,
        horizon: int,
        split: str,
        n_windows: int,
        levels: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Return cached quantiles shape (n_windows, Q, H) if a hit, else None."""
        path = self._path(dataset, agent, seq_len, horizon, split)
        if not path.exists():
            return None
        d = np.load(path)
        cached_levels = d["levels"]
        if cached_levels.shape != levels.shape or not np.allclose(cached_levels, levels):
            return None
        cached_q = d["quantiles"]
        if cached_q.shape[0] < n_windows:
            return None
        return cached_q[:n_windows].astype(np.float32, copy=False)

    def save(
        self,
        *,
        dataset: str,
        agent: str,
        seq_len: int,
        horizon: int,
        split: str,
        quantiles: np.ndarray,
        levels: np.ndarray,
    ) -> Path:
        path = self._path(dataset, agent, seq_len, horizon, split)
        np.savez_compressed(
            path,
            quantiles=np.asarray(quantiles, dtype=np.float32),
            levels=np.asarray(levels, dtype=np.float32),
        )
        return path


def cached_forecast(
    *,
    cache: ForecastCache,
    dataset: str,
    agent_name: str,
    seq_len: int,
    horizon: int,
    split: str,
    histories: np.ndarray,
    levels: np.ndarray,
    compute_fn: Callable[[np.ndarray, int, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Return cached forecasts or compute + cache.

    compute_fn signature: (histories, horizon, levels) -> quantiles (N, Q, H).
    """
    hit = cache.load(
        dataset=dataset,
        agent=agent_name,
        seq_len=seq_len,
        horizon=horizon,
        split=split,
        n_windows=len(histories),
        levels=levels,
    )
    if hit is not None:
        return hit
    out = compute_fn(histories, horizon, levels)
    out = np.asarray(out, dtype=np.float32)
    cache.save(
        dataset=dataset,
        agent=agent_name,
        seq_len=seq_len,
        horizon=horizon,
        split=split,
        quantiles=out,
        levels=levels,
    )
    return out
