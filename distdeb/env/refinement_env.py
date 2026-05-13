"""Sequential refinement RL environment.

One episode = one forecasting window. The policy sees a state and chooses
to invoke an additional agent (cached quantile forecast retrieved instantly)
or HALT, in which case the current equal-weight ensemble of called agents
is the final forecast. Reward is terminal CRPS plus a per-step cost penalty.

This is the MDP from DESIGN.md S3 (v2 sequential refinement).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..eval.metrics import quantile_crps


def window_features(history: np.ndarray) -> np.ndarray:
    """Per-window summary stats (per-window normalized so different datasets are comparable).

    Returns shape (10,):
      mean, std, min, max, range, q25, q75, skew, autocorr_lag1, trend_slope
    All computed on z-scored history (the loader has already done dataset-level
    z-scoring; this is an additional per-window normalization).
    """
    x = np.asarray(history, dtype=np.float32).ravel()
    if x.size == 0:
        return np.zeros(10, dtype=np.float32)
    mu = float(x.mean())
    sd = float(x.std()) + 1e-8
    z = (x - mu) / sd
    q25, q75 = np.quantile(z, [0.25, 0.75])
    skew = float(np.mean(z ** 3))
    # lag-1 autocorr
    if x.size > 1:
        ac = float(np.corrcoef(z[:-1], z[1:])[0, 1])
        if not np.isfinite(ac):
            ac = 0.0
    else:
        ac = 0.0
    # Linear trend slope (in z-units per step)
    t = np.arange(x.size, dtype=np.float32)
    if x.size > 1:
        slope = float(np.polyfit(t, z, 1)[0])
    else:
        slope = 0.0
    return np.array(
        [mu, sd, float(z.min()), float(z.max()), float(z.max() - z.min()),
         float(q25), float(q75), skew, ac, slope],
        dtype=np.float32,
    )


class RefinementEnv:
    """Sequential refinement env over pre-cached agent forecasts.

    Constructor packs all windows and forecasts into in-memory arrays — episodes
    are then near-free to roll out, so PPO training is GPU-bound only on the
    policy forward/backward.

    Conventions:
      - actions: 0..N_agents-1 = call agent i. N_agents = HALT.
      - already-called agents are masked out (action's logit set to -inf).
      - empty ensemble at HALT is penalized via terminal reward = -1.0.
    """

    def __init__(
        self,
        histories: np.ndarray,            # (N, L) — lookback windows
        targets: np.ndarray,              # (N, H) — true horizons (z-scored)
        agent_quantiles: Dict[str, np.ndarray],  # name -> (N, Q, H)
        levels: np.ndarray,               # (Q,) quantile levels
        dataset_ids: Optional[np.ndarray] = None,  # (N,) per-window dataset id
        n_datasets: Optional[int] = None,  # required if include_dataset_id=True
        agent_costs: Optional[Dict[str, float]] = None,
        cost_weight: float = 0.001,
        feature_fn=window_features,
        include_dataset_id: bool = False,  # adds one-hot dataset id to state
    ):
        self.histories = np.asarray(histories, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.levels = np.asarray(levels, dtype=np.float32)
        self.agent_names = list(agent_quantiles.keys())
        self.N_agents = len(self.agent_names)
        self.quantiles = {n: np.asarray(q, dtype=np.float32) for n, q in agent_quantiles.items()}
        self.dataset_ids = (
            np.zeros(len(self.histories), dtype=np.int32) if dataset_ids is None else np.asarray(dataset_ids)
        )
        if include_dataset_id and n_datasets is None:
            n_datasets = int(self.dataset_ids.max()) + 1
        self.include_dataset_id = include_dataset_id
        self.n_datasets = n_datasets or 0
        self.agent_costs = (
            {n: 1.0 for n in self.agent_names} if agent_costs is None else dict(agent_costs)
        )
        self.cost_weight = float(cost_weight)
        self.feature_fn = feature_fn

        self.N, self.L = self.histories.shape
        self.Q, self.H = self.levels.shape[0], self.targets.shape[1]
        for n, q in self.quantiles.items():
            assert q.shape == (self.N, self.Q, self.H), (
                f"agent {n} quantiles shape {q.shape} != expected {(self.N, self.Q, self.H)}"
            )

        # Precompute per-window features so they're cheap to fetch.
        self._feat_dim = len(self.feature_fn(self.histories[0]))
        self._features = np.stack(
            [self.feature_fn(self.histories[i]) for i in range(self.N)], axis=0
        ).astype(np.float32)

        # State layout:
        #   window_features (feat_dim)
        #   [optional] dataset_id one-hot (n_datasets)
        #   ensemble_quantiles (Q*H)
        #   width (1)
        #   called_mask (N_agents)
        #   n_calls (1)
        extra_ds = self.n_datasets if self.include_dataset_id else 0
        self.state_dim = self._feat_dim + extra_ds + self.Q * self.H + 1 + self.N_agents + 1
        self.n_actions = self.N_agents + 1  # +1 for HALT
        self.halt_action = self.N_agents

        self._window_idx: int = -1
        self._called: List[int] = []

    def reset(self, window_idx: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if window_idx is None:
            rng = rng if rng is not None else np.random.default_rng()
            self._window_idx = int(rng.integers(0, self.N))
        else:
            self._window_idx = int(window_idx)
        self._called = []
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        info: Dict = {"called": list(self._called), "halt": False, "invalid": False}
        if action == self.halt_action:
            reward = self._terminal_reward()
            info["halt"] = True
            return self._get_state(), reward, True, info

        if action < 0 or action >= self.N_agents:
            raise ValueError(f"action {action} out of range [0, {self.N_agents}]")

        if action in self._called:
            info["invalid"] = True
            # Invalid: small penalty, no state change.
            return self._get_state(), -0.1, False, info

        self._called.append(action)
        step_cost = self.cost_weight * self.agent_costs[self.agent_names[action]]
        reward = -step_cost

        if len(self._called) == self.N_agents:
            # Forced halt: all agents used.
            reward += self._terminal_reward()
            return self._get_state(), reward, True, info

        return self._get_state(), reward, False, info

    def valid_action_mask(self) -> np.ndarray:
        """1 = valid, 0 = invalid (already called). HALT (last action) is always valid."""
        mask = np.ones(self.n_actions, dtype=np.float32)
        for a in self._called:
            mask[a] = 0.0
        return mask

    # ---------- internal ----------

    def _get_state(self) -> np.ndarray:
        feat = self._features[self._window_idx]
        parts = [feat]
        if self.include_dataset_id:
            ds_onehot = np.zeros(self.n_datasets, dtype=np.float32)
            ds_id = int(self.dataset_ids[self._window_idx])
            if 0 <= ds_id < self.n_datasets:
                ds_onehot[ds_id] = 1.0
            parts.append(ds_onehot)
        if self._called:
            stacked = np.stack(
                [self.quantiles[self.agent_names[a]][self._window_idx] for a in self._called],
                axis=0,
            )
            ensemble = stacked.mean(axis=0)  # (Q, H)
            lo = ensemble[0]
            hi = ensemble[-1]
            width = float(np.mean(hi - lo))
        else:
            ensemble = np.zeros((self.Q, self.H), dtype=np.float32)
            width = 0.0
        mask = np.zeros(self.N_agents, dtype=np.float32)
        for a in self._called:
            mask[a] = 1.0
        n_calls = float(len(self._called))
        parts.extend([
            ensemble.flatten(),
            np.array([width], dtype=np.float32),
            mask,
            np.array([n_calls], dtype=np.float32),
        ])
        return np.concatenate(parts).astype(np.float32)

    def _terminal_reward(self) -> float:
        """Negative CRPS of the current ensemble vs. the target for this window."""
        if not self._called:
            # Halt with no calls → maximally bad terminal reward.
            return -1.0
        stacked = np.stack(
            [self.quantiles[self.agent_names[a]][self._window_idx] for a in self._called],
            axis=0,
        )
        ensemble = stacked.mean(axis=0)  # (Q, H)
        # quantile_crps expects (Q, ..., H) with target matching trailing axes.
        target = self.targets[self._window_idx]  # (H,)
        crps = quantile_crps(ensemble.reshape(self.Q, 1, -1), self.levels, target.reshape(1, -1))
        return float(-crps)
