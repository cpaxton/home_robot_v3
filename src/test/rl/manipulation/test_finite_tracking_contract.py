"""A stopped, scored deadline is terminal; a generic collection cut is not."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from emet.rl.manipulation.multi_critic_ppo import grouped_gae
from emet.rl.manipulation.whole_body_actor import MultiCriticPolicy
from emet.rl.manipulation.whole_body_training import collect_rollout


@pytest.mark.sim
@pytest.mark.parametrize("kind", ["general", "coupled"])
def test_physical_tracking_deadline_cannot_bootstrap_stopped_state(kind, monkeypatch):
    if kind == "general":
        from emet.rl.manipulation.whole_body_training import make_environment

        env = make_environment(
            "configs/benchmarks/mars_whole_body_v2_development.json",
            basic_cases=Path("configs/benchmarks/mars_whole_body_basic_learning_v1.json"),
        )
        native = env.environments["general"].unwrapped
    else:
        from emet.rl.manipulation.coupled_tracking import CoupledTrackingEnv

        env = CoupledTrackingEnv("innate_mars")
        native = env.unwrapped
    try:
        original_reset = env.reset

        def bounded_reset(*args, **kwargs):
            result = original_reset(*args, **kwargs)
            # Mastery restores its scoped configuration on every reset.
            native.config = replace(native.config, horizon=native.profile.dt)
            return result

        monkeypatch.setattr(env, "reset", bounded_reset)
        obs, _ = env.reset(seed=90)
        _, _, terminated, truncated, info = env.step(np.zeros(8))
        assert terminated and not truncated and native.control.stopped
        assert info["native_truncated"] and not info["native_terminated"]
        assert info["episode_contract"] == "finite_tracking_deadline_v1"
        assert not info["primitive_success"]  # no success relabeling
        model = MultiCriticPolicy(hidden=16)
        monkeypatch.setattr(model, "values", lambda observation: torch.full((len(observation["history"]), 3), 100.0))
        obs, _ = env.reset(seed=90)
        rollout, _ = collect_rollout(env, model, obs, 2)
        assert rollout["terminated"].all() and rollout["episode_end"].all()
        _, returns = grouped_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["next_values"],
            rollout["terminated"],
            rollout["episode_end"],
            rollout["ticks"],
        )
        np.testing.assert_allclose(returns, rollout["rewards"], atol=1e-5)
    finally:
        env.close()
