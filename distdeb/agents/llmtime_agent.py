"""Qwen-LLMTime agent.

LLMTime (Gruver et al., NeurIPS 2023) — zero-shot TS forecasting by treating
a numeric series as a textual sequence, prompting an LLM to continue it, and
extracting an empirical quantile forecast from many samples.

We wrap Qwen2.5 (base, not instruct) as the panel's LLM. Key recipe choices
follow the LLMTime paper:
  - Per-window scaling so values fit a useful magnitude range (~5x).
  - Space-between-digit tokenization to force per-digit tokens through Qwen's
    BPE tokenizer (without spaces, "1.23" might compress into a single token,
    ruining the autoregressive arithmetic).
  - Multi-sample generation (n_samples=20), quantiles from empirical samples.

Reference:
  Gruver et al., "Large Language Models Are Zero-Shot Time Series Forecasters",
  NeurIPS 2023. https://arxiv.org/abs/2310.07820
  Repo: https://github.com/ngruver/llmtime
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from .base import Agent, Forecast


_DEFAULT_MODEL = "Qwen/Qwen2.5-7B"


class QwenLLMTimeAgent(Agent):
    """LLM-as-forecaster agent.

    nominal_cost is 5x by default to reflect that this is a 7B LLM vs Chronos's
    ~200M foundation model. The RL orchestrator will use this for cost-aware
    routing decisions.
    """

    nominal_cost = 5.0

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        n_samples: int = 20,
        temperature: float = 0.7,
        top_p: float = 0.9,
        precision: int = 2,
        pre_scale: float = 5.0,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        max_new_token_factor: int = 8,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device,
        )
        self.model.eval()

        self.device = device
        self.model_id = model_id
        self.n_samples = n_samples
        self.temperature = temperature
        self.top_p = top_p
        self.precision = precision
        self.pre_scale = pre_scale
        self.max_new_token_factor = max_new_token_factor

        short = model_id.split("/")[-1].replace(".", "").lower()
        # Distinct cache key per model + n_samples + temperature.
        self.name = f"llmtime_{short}_n{n_samples}_t{int(temperature * 10)}"

    def forecast(
        self,
        history: np.ndarray,
        horizon: int,
        levels: np.ndarray,
        context: Optional[dict] = None,
    ) -> Forecast:
        import torch

        history = np.asarray(history, dtype=np.float32)
        if history.ndim == 2:
            history = history.squeeze(-1)

        abs_max = float(np.abs(history).max()) + 1e-6
        history_str = self._encode_history(history, abs_max)
        prompt = (
            "Time series forecast. Continue the sequence with exactly "
            f"{horizon} more comma-separated values:\n"
            f"{history_str} ,"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        max_new = horizon * self.max_new_token_factor + 32

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                num_return_sequences=self.n_samples,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        samples = []
        for i in range(self.n_samples):
            completion = self.tokenizer.decode(
                outputs[i, input_length:], skip_special_tokens=True
            )
            samples.append(self._parse_completion(completion, horizon, abs_max, self.pre_scale))
        samples = np.stack(samples)  # (n_samples, horizon)
        quantiles = np.quantile(samples, levels, axis=0).astype(np.float32)
        return Forecast(
            quantiles=quantiles,
            levels=np.asarray(levels),
            agent_name=self.name,
            cost=self.nominal_cost,
        )

    def forecast_batch(
        self,
        histories: np.ndarray,
        horizon: int,
        levels: np.ndarray,
        batch_size: int = 1,  # unused; LLM sampling parallelism is internal to .forecast
    ) -> np.ndarray:
        histories = np.asarray(histories, dtype=np.float32)
        assert histories.ndim == 2, f"expected (N, L), got {histories.shape}"
        N = histories.shape[0]
        Q = len(levels)
        out = np.empty((N, Q, horizon), dtype=np.float32)
        for i in range(N):
            fc = self.forecast(histories[i], horizon, levels)
            out[i] = fc.quantiles
        return out

    def _format_number(self, x: float) -> str:
        s = f"{x:.{self.precision}f}"
        # Insert a space between every char so BPE tokenizes per-digit.
        return " ".join(s)

    def _encode_history(self, history: np.ndarray, scale: float) -> str:
        scaled = history / scale * self.pre_scale
        return " , ".join(self._format_number(float(x)) for x in scaled)

    @staticmethod
    def _parse_completion(text: str, horizon: int, scale: float, pre_scale: float) -> np.ndarray:
        """Extract horizon numbers from a LLM completion.

        The completion has digits separated by spaces, numbers separated by
        commas. We strip whitespace, then capture float-like patterns.
        Malformed outputs (truncated, garbled) are padded with the last
        successfully parsed value.
        """
        compact = re.sub(r"\s+", "", text)
        nums = re.findall(r"-?\d+(?:\.\d+)?", compact)
        values = []
        for n in nums[:horizon]:
            try:
                v = float(n)
                values.append(v / pre_scale * scale)
            except ValueError:
                pass
        if not values:
            return np.zeros(horizon, dtype=np.float32)
        while len(values) < horizon:
            values.append(values[-1])
        return np.asarray(values[:horizon], dtype=np.float32)
