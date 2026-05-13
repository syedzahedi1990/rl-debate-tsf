"""Minimal PPO trainer for the RefinementEnv.

Single-thread rollouts (env is in-memory and near-free), clipped surrogate
objective, GAE advantage, action masking via -inf logit additive on
invalid actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass
class PPOConfig:
    n_iters: int = 200
    n_episodes_per_iter: int = 256
    n_epochs: int = 4
    minibatch_size: int = 512
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 0


class PPOTrainer:
    def __init__(self, env, policy, config: Optional[PPOConfig] = None, device: str = "cpu"):
        self.env = env
        self.policy = policy.to(device)
        self.config = config or PPOConfig()
        self.device = device
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.lr)
        self.rng = np.random.default_rng(self.config.seed)

    def _act(self, state: np.ndarray, mask: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.policy(s)
            logits = logits + torch.log(m + 1e-10)  # mask invalid actions to -inf
            if deterministic:
                action = int(torch.argmax(logits, dim=-1).item())
                logp = float(F.log_softmax(logits, dim=-1)[0, action].item())
            else:
                dist = Categorical(logits=logits)
                a = dist.sample()
                action = int(a.item())
                logp = float(dist.log_prob(a).item())
        return action, logp, float(value.item())

    def _collect_rollouts(self) -> dict:
        states, actions, masks, logps, rewards, values, dones, ep_returns, ep_lens = (
            [], [], [], [], [], [], [], [], []
        )
        for _ in range(self.config.n_episodes_per_iter):
            state = self.env.reset(rng=self.rng)
            ep_return = 0.0
            steps_in_ep = 0
            while True:
                mask = self.env.valid_action_mask()
                action, logp, value = self._act(state, mask)
                next_state, reward, done, _info = self.env.step(action)
                states.append(state)
                actions.append(action)
                masks.append(mask)
                logps.append(logp)
                values.append(value)
                rewards.append(reward)
                dones.append(done)
                state = next_state
                ep_return += reward
                steps_in_ep += 1
                if done:
                    break
            ep_returns.append(ep_return)
            ep_lens.append(steps_in_ep)

        # Bootstrap final value (always 0 since all our episodes terminate).
        advantages = np.zeros(len(rewards), dtype=np.float32)
        returns = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            nv = 0.0 if dones[t] else next_value
            delta = rewards[t] + self.config.gamma * nv - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (0.0 if dones[t] else gae)
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_value = values[t]

        return {
            "states": np.asarray(states, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.int64),
            "masks": np.asarray(masks, dtype=np.float32),
            "logps": np.asarray(logps, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "values": np.asarray(values, dtype=np.float32),
            "advantages": advantages,
            "returns": returns,
            "ep_returns": np.asarray(ep_returns, dtype=np.float32),
            "ep_lens": np.asarray(ep_lens, dtype=np.float32),
        }

    def _update(self, batch: dict) -> dict:
        states = torch.as_tensor(batch["states"], device=self.device)
        actions = torch.as_tensor(batch["actions"], device=self.device)
        masks = torch.as_tensor(batch["masks"], device=self.device)
        old_logps = torch.as_tensor(batch["logps"], device=self.device)
        advantages = torch.as_tensor(batch["advantages"], device=self.device)
        returns = torch.as_tensor(batch["returns"], device=self.device)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = states.shape[0]
        losses = []
        kls = []
        for _ in range(self.config.n_epochs):
            idx = torch.randperm(N, device=self.device)
            for start in range(0, N, self.config.minibatch_size):
                end = min(start + self.config.minibatch_size, N)
                mb = idx[start:end]
                logits, values = self.policy(states[mb])
                logits = logits + torch.log(masks[mb] + 1e-10)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(actions[mb])
                ratio = torch.exp(logp - old_logps[mb])
                surr1 = ratio * advantages[mb]
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * advantages[mb]
                pol_loss = -torch.min(surr1, surr2).mean()
                val_loss = F.mse_loss(values, returns[mb])
                ent = dist.entropy().mean()
                loss = pol_loss + self.config.value_coef * val_loss - self.config.entropy_coef * ent
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                losses.append(float(loss.item()))
                with torch.no_grad():
                    kls.append(float((old_logps[mb] - logp).mean().item()))
        return {"loss": float(np.mean(losses)), "kl": float(np.mean(kls))}

    def train(self, log_every: int = 10) -> List[dict]:
        history = []
        for it in range(self.config.n_iters):
            batch = self._collect_rollouts()
            stats = self._update(batch)
            rec = {
                "iter": it,
                "mean_return": float(batch["ep_returns"].mean()),
                "mean_ep_len": float(batch["ep_lens"].mean()),
                **stats,
            }
            history.append(rec)
            if (it + 1) % log_every == 0 or it == 0:
                print(
                    f"  iter {it + 1:4d}/{self.config.n_iters}  "
                    f"return={rec['mean_return']:+.4f}  "
                    f"ep_len={rec['mean_ep_len']:.2f}  "
                    f"loss={rec['loss']:.4f}  "
                    f"kl={rec['kl']:+.4f}"
                )
        return history

    def evaluate(self, env, n_windows: int, deterministic: bool = True) -> dict:
        """Roll out one episode per window 0..n_windows-1 deterministically.

        Returns aggregate metrics and per-window arrays.
        """
        ensembles = []  # (n_windows, Q, H)
        chosen = []    # list of called-agents per window
        for w in range(n_windows):
            state = env.reset(window_idx=w)
            called = []
            while True:
                mask = env.valid_action_mask()
                action, _logp, _v = self._act(state, mask, deterministic=deterministic)
                next_state, _r, done, _info = env.step(action)
                state = next_state
                if action != env.halt_action:
                    called.append(action)
                if done:
                    break
            chosen.append(called)
            if called:
                stacked = np.stack(
                    [env.quantiles[env.agent_names[a]][w] for a in called], axis=0
                )
                ensembles.append(stacked.mean(axis=0))
            else:
                # Fallback: equal-weight all agents (shouldn't happen with a trained policy).
                stacked = np.stack(
                    [env.quantiles[n][w] for n in env.agent_names], axis=0
                )
                ensembles.append(stacked.mean(axis=0))
        ensembles_arr = np.stack(ensembles, axis=0)  # (n_windows, Q, H)
        return {
            "ensembles": ensembles_arr,
            "chosen": chosen,
            "mean_n_calls": float(np.mean([len(c) for c in chosen])),
        }
