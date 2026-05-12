"""Moirai agent wrapper.

Moirai (Liu et al., ICML 2024) is Salesforce's universal forecasting
transformer: any-variate masked attention, patch-based input, multi-quantile
sample head. Architecture is substantively different from both Chronos
(encoder-decoder T5 + value tokenization) and LLM-based forecasters
(autoregressive digit sampling) -> good panel diversity.

Reference:
  Liu et al., "Unified Training of Universal Time Series Forecasting
  Transformers", ICML 2024. https://arxiv.org/abs/2402.02592
  Repo: https://github.com/SalesforceAIResearch/uni2ts
  Weights: Salesforce/moirai-1.1-R-base (91M params)

We use the direct PyTorch forward interface (not the GluonTS predictor) to
avoid the dataset-iteration overhead and to keep the wrapper minimal.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Agent, Forecast


class MoiraiAgent(Agent):
    nominal_cost = 1.0

    def __init__(
        self,
        model_id: str = "Salesforce/moirai-1.1-R-base",
        num_samples: int = 100,
        patch_size: int = 32,
        device: Optional[str] = None,
        name_suffix: str = "",
    ):
        import torch
        from uni2ts.model.moirai import MoiraiModule

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.module = MoiraiModule.from_pretrained(model_id)
        self.module.to(device)
        self.module.eval()
        self.device = device
        self.model_id = model_id
        self.num_samples = num_samples
        self.default_patch_size = patch_size
        self.name = f"moirai{name_suffix}"

    def forecast(
        self,
        history: np.ndarray,
        horizon: int,
        levels: np.ndarray,
        context: Optional[dict] = None,
    ) -> Forecast:
        history = np.asarray(history, dtype=np.float32)
        if history.ndim == 2:
            qs = np.stack(
                [self._forecast_uni(history[:, v], horizon, levels) for v in range(history.shape[1])],
                axis=-1,
            )
        else:
            qs = self._forecast_uni(history, horizon, levels)
        return Forecast(quantiles=qs, levels=np.asarray(levels), agent_name=self.name, cost=self.nominal_cost)

    def forecast_batch(
        self,
        histories: np.ndarray,
        horizon: int,
        levels: np.ndarray,
        batch_size: int = 32,
    ) -> np.ndarray:
        import torch
        from uni2ts.model.moirai import MoiraiForecast

        histories = np.asarray(histories, dtype=np.float32)
        assert histories.ndim == 2, f"expected (N, L), got {histories.shape}"
        N, L = histories.shape
        Q = len(levels)

        patch_size = self._pick_patch_size(L)

        forecaster = MoiraiForecast(
            module=self.module,
            prediction_length=horizon,
            context_length=L,
            patch_size=patch_size,
            num_samples=self.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        forecaster.to(self.device)
        forecaster.eval()

        out = np.empty((N, Q, horizon), dtype=np.float32)
        for i in range(0, N, batch_size):
            batch = histories[i : i + batch_size]
            B = batch.shape[0]
            past_target = torch.tensor(batch, dtype=torch.float32).unsqueeze(-1).to(self.device)
            past_observed = torch.ones_like(past_target, dtype=torch.bool)
            past_is_pad = torch.zeros(B, L, dtype=torch.bool).to(self.device)
            with torch.inference_mode():
                samples = forecaster(
                    past_target=past_target,
                    past_observed_target=past_observed,
                    past_is_pad=past_is_pad,
                )
            arr = samples.detach().to("cpu").float().numpy()
            if arr.ndim == 4:
                arr = arr.squeeze(-1)  # (B, num_samples, horizon)
            for j in range(B):
                out[i + j] = np.quantile(arr[j], levels, axis=0).astype(np.float32)
        return out

    def _forecast_uni(self, x: np.ndarray, horizon: int, levels: np.ndarray) -> np.ndarray:
        return self.forecast_batch(x[None, :], horizon, levels)[0]

    def _pick_patch_size(self, context_len: int) -> int:
        """Moirai requires context_len % patch_size == 0. Try our default first,
        then walk through common alternatives."""
        if context_len % self.default_patch_size == 0:
            return self.default_patch_size
        for ps in (16, 8, 32, 64, 4):
            if context_len % ps == 0:
                return ps
        raise ValueError(f"No supported patch size divides context_len={context_len}")
