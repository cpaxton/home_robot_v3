"""Critic fitting budgets, actor isolation, and honest fit diagnostics."""

import copy

import numpy as np
import pytest
import torch

from emet.rl.manipulation.multi_critic_ppo import MultiCriticPPO, PPOConfig, value_fit_statistics
from emet.rl.manipulation.observation_layout import history_layout, preview_layout
from emet.rl.manipulation.profiles import MARS
from emet.rl.manipulation.whole_body_actor import MultiCriticPolicy


def rollout(model, count=16, *, violate_kl=False):
    fields, width = history_layout(MARS, retention_context=True)
    pf, _ = preview_layout()
    history = torch.zeros(count, 20, width)
    history[..., fields["hand_pose"]][..., 3:] = torch.tensor([1.0, 0, 0, 1, 0, 0])
    history[..., fields["hand_mask"]] = 1
    history[..., fields["base_mask"]] = 1
    preview = torch.zeros(count, 7, 18)
    preview[..., pf["hand_error"]][..., 3:] = torch.tensor([1.0, 0, 0, 1, 0, 0])
    preview[..., pf["hand_error"]][..., 0] = 0.01
    preview[..., pf["hand_mask"]] = 1
    preview[..., pf["base_mask"]] = 1
    obs = {"history": history, "preview": preview, "motion": torch.zeros(count, 9), "critic": torch.zeros(count, 34)}
    with torch.no_grad():
        actions = model.distribution(obs).sample()
        log, _, values = model.evaluate(obs, actions)
    return {
        "observations": {k: v.numpy() for k, v in obs.items()},
        "actions": actions.numpy(),
        "log_prob": log.numpy() - (2 if violate_kl else 0),
        "rewards": np.full((count, 3), 0.5, np.float32),
        "values": values.numpy(),
        "next_values": np.zeros((count, 3), np.float32),
        "terminated": np.ones(count, bool),
        "episode_end": np.ones(count, bool),
        "ticks": np.ones(count),
    }


@pytest.mark.parametrize("critic_epochs", [0, 4])
def test_actor_kl_stop_does_not_starve_independent_critics(critic_epochs):
    torch.manual_seed(19)
    model = MultiCriticPolicy(hidden=16)
    data = rollout(model, violate_kl=True)
    actor = {k: v.clone() for k, v in model.actor.state_dict().items()}
    std = model.log_std.detach().clone()
    before = {k: v.clone() for k, v in model.critics.state_dict().items()}
    learner = MultiCriticPPO(
        model, PPOConfig(epochs=2, critic_epochs=critic_epochs, batch_size=8, critic_learning_rate=0.001)
    )
    result = learner.update(data)
    assert result["actor_kl_early_stop"] and result["optimizer_steps"] == 0
    assert result["critic_optimizer_steps"] == critic_epochs * 2
    for k, v in model.actor.state_dict().items():
        torch.testing.assert_close(v, actor[k], rtol=0, atol=0)
    torch.testing.assert_close(model.log_std, std, rtol=0, atol=0)
    # Adam momentum/step state must never be created for frozen actor variables.
    assert all(p not in learner.optimizer.state for p in [*model.actor.parameters(), model.log_std])
    changed = any(not torch.equal(before[k], v) for k, v in model.critics.state_dict().items())
    assert changed == (critic_epochs > 0)
    if critic_epochs:
        assert sum(result["critic_gae_target_fit_after"]["rmse"]) < sum(result["critic_gae_target_fit_before"]["rmse"])


def test_independent_critic_budget_preserves_same_rollout_actor_updates_and_rng():
    torch.manual_seed(23)
    source = MultiCriticPolicy(hidden=16)
    data = rollout(source)
    outcomes = []
    for epochs in (0, 3):
        policy = copy.deepcopy(source)
        learner = MultiCriticPPO(policy, PPOConfig(epochs=2, critic_epochs=epochs, batch_size=8, target_kl=1.0))
        torch.manual_seed(101)
        result = learner.update(data)
        outcomes.append((policy, torch.get_rng_state().clone(), result))
    first, second = outcomes
    for k, v in first[0].actor.state_dict().items():
        torch.testing.assert_close(v, second[0].actor.state_dict()[k], rtol=0, atol=0)
    torch.testing.assert_close(first[0].log_std, second[0].log_std, rtol=0, atol=0)
    torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
    assert first[2]["optimizer_steps"] == second[2]["optimizer_steps"] == 4
    assert first[2]["critic_optimizer_steps"] == 4 and second[2]["critic_optimizer_steps"] == 6


def test_fit_statistics_distinguish_bias_and_undefined_constant_target_variance():
    targets = np.c_[np.arange(4), np.ones(4), np.arange(4)]
    stats = value_fit_statistics(targets, targets + [2, 0, 0])
    assert stats["rmse"] == [2, 0, 0] and stats["bias"] == [2, 0, 0]
    assert stats["explained_variance"] == [1, None, 1]  # EV alone hides constant bias
    assert stats["samples"] == 4
    for bad in (np.zeros((4, 2)), np.full((4, 3), np.nan)):
        with pytest.raises(ValueError):
            value_fit_statistics(targets, bad)


@pytest.mark.parametrize("value", [-1, True, 0.5, float("nan")])
def test_critic_epoch_budget_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="critic epochs"):
        PPOConfig(critic_epochs=value)
