"""Unit tests for RefinementEnv."""

import numpy as np
import pytest

from distdeb.env.refinement_env import RefinementEnv, window_features


@pytest.fixture
def toy_env():
    rng = np.random.default_rng(0)
    N, L, Q, H = 8, 32, 5, 16
    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    histories = rng.normal(size=(N, L)).astype(np.float32)
    targets = rng.normal(size=(N, H)).astype(np.float32)
    agent_quantiles = {
        "a": rng.normal(size=(N, Q, H)).astype(np.float32),
        "b": rng.normal(size=(N, Q, H)).astype(np.float32),
        "c": rng.normal(size=(N, Q, H)).astype(np.float32),
    }
    return RefinementEnv(
        histories=histories,
        targets=targets,
        agent_quantiles=agent_quantiles,
        levels=levels,
        cost_weight=0.01,
    )


def test_window_features_shape():
    x = np.arange(20, dtype=np.float32)
    feat = window_features(x)
    assert feat.shape == (10,)
    assert np.all(np.isfinite(feat))


def test_window_features_empty():
    feat = window_features(np.array([], dtype=np.float32))
    assert feat.shape == (10,)


def test_reset_returns_state(toy_env):
    state = toy_env.reset(window_idx=3)
    assert state.shape == (toy_env.state_dim,)
    assert np.all(np.isfinite(state))


def test_halt_with_no_calls_gives_terminal_penalty(toy_env):
    toy_env.reset(window_idx=0)
    _, reward, done, info = toy_env.step(toy_env.halt_action)
    assert done
    assert info["halt"]
    assert reward == -1.0


def test_call_then_halt_terminal_reward_is_negative_crps(toy_env):
    toy_env.reset(window_idx=0)
    toy_env.step(0)  # call agent 'a'
    _, reward, done, _ = toy_env.step(toy_env.halt_action)
    assert done
    # CRPS is non-negative, so terminal reward = -CRPS <= 0.
    assert reward <= 0


def test_calling_same_agent_twice_is_invalid(toy_env):
    toy_env.reset(window_idx=0)
    toy_env.step(0)
    _, reward, done, info = toy_env.step(0)
    assert not done
    assert info["invalid"]
    assert reward == -0.1


def test_calling_all_agents_force_terminal(toy_env):
    toy_env.reset(window_idx=0)
    toy_env.step(0)
    toy_env.step(1)
    _, reward, done, info = toy_env.step(2)
    assert done
    # final step gets both per-step cost AND terminal CRPS reward
    assert reward < 0


def test_valid_action_mask(toy_env):
    toy_env.reset(window_idx=0)
    mask = toy_env.valid_action_mask()
    assert mask.shape == (toy_env.n_actions,)
    assert mask.sum() == toy_env.n_actions  # all valid at start
    toy_env.step(1)  # call 'b'
    mask = toy_env.valid_action_mask()
    assert mask[1] == 0.0  # b masked
    assert mask[toy_env.halt_action] == 1.0  # HALT always valid


def test_state_dim_consistent(toy_env):
    feat_dim = 10
    Q, H, N_agents = 5, 16, 3
    expected = feat_dim + Q * H + 1 + N_agents + 1
    assert toy_env.state_dim == expected
