"""Dataset loaders.

For the pilot we keep loading dead-simple: read CSVs directly from the
TSLib-format `dataset/` directory and yield (lookback, horizon) windows in the
canonical Informer-suite splits. This avoids pulling TSLib's argparse-driven
data_factory in for a smoke test.

Splits (Informer convention, hard-coded in TSLib data_loader.py):
  ETTh: 12mo train, 4mo val, 4mo test (boundaries at indices 12*30*24 and 16*30*24)
  ETTm: same boundaries but in 15-min units (12*30*24*4, 16*30*24*4)

z-scoring uses train-set mean/std. Metrics on the Informer suite are reported
on the z-scored scale by convention.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
import pandas as pd


@dataclass
class StandardizedSeries:
    data: np.ndarray  # (T, V)
    mean: np.ndarray  # (V,)
    std: np.ndarray  # (V,)
    train_end: int
    val_end: int

    def unscale(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def _ett_boundaries(freq: str) -> Tuple[int, int]:
    if freq == "hour":
        return 12 * 30 * 24, 16 * 30 * 24
    if freq == "minute":
        return 12 * 30 * 24 * 4, 16 * 30 * 24 * 4
    raise ValueError(f"unknown freq {freq!r}")


def load_etth1_windows(
    csv_path: str,
    seq_len: int = 96,
    pred_len: int = 96,
    features: str = "S",
    target: str = "OT",
) -> StandardizedSeries:
    """Load ETTh1 (or any ETTh-format file) and return a standardized series.

    features='S' → univariate target series; 'M' → multivariate (all cols).
    """
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    if features == "S":
        arr = df[[target]].values.astype(np.float32)
    elif features == "M":
        arr = df.values.astype(np.float32)
    else:
        raise ValueError(f"features must be 'S' or 'M', got {features!r}")

    train_end, val_end = _ett_boundaries("hour")
    mean = arr[:train_end].mean(axis=0)
    std = arr[:train_end].std(axis=0) + 1e-8
    arr_z = (arr - mean) / std

    return StandardizedSeries(
        data=arr_z, mean=mean, std=std, train_end=train_end, val_end=val_end
    )


def iter_test_windows(
    series: StandardizedSeries,
    seq_len: int,
    pred_len: int,
    stride: int = 1,
    limit: int | None = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (history, target) pairs from the test split.

    history: (seq_len, V); target: (pred_len, V). Stride controls how many
    windows. limit caps the total number of windows yielded.
    """
    data = series.data
    test_start = series.val_end
    end = len(data) - pred_len
    n = 0
    for i in range(test_start, end, stride):
        if i - seq_len < 0:
            continue
        history = data[i - seq_len : i]
        target = data[i : i + pred_len]
        yield history, target
        n += 1
        if limit is not None and n >= limit:
            break
