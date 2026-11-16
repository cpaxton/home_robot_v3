# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Reward priorities must survive the actor's advantage transformation."""

import numpy as np
import pytest
import torch
from torch import nn

from emet.rl.manipulation.multi_critic_ppo import MultiCriticPPO, PPOConfig, combine_advantages, grouped_gae


def test_small_motion_penalty_does_not_cancel_tracking_after_normalization():
    tracking = torch.tensor([-3.0, -1.0, 1.0, 3.0])
    advantages = torch.stack([tracking, -0.01 * tracking, torch.zeros_like(tracking)], dim=-1)
    combined, _ = combine_advantages(advantages)
    expected = tracking / tracking.std(unbiased=False)
    torch.testing.assert_close(combined, expected)
    old, _ = combine_advantages(advantages, normalization="per_head")
    torch.testing.assert_close(old, torch.zeros_like(tracking), atol=2e-6, rtol=0)


def test_shared_normalization_matches_weighted_total_and_preserves_relative_coefficients():
    torch.manual_seed(90)
    advantages = torch.randn(128, 3)
    weights = torch.tensor([2.0, 0.01, 0.1])
    combined, components = combine_advantages(advantages, weights)
    torch.testing.assert_close(components.sum(-1), combined)
    total = advantages @ weights
    torch.testing.assert_close(combined, (total - total.mean()) / total.std(unbiased=False))
    changed, _ = combine_advantages(advantages * torch.tensor([1.0, 1000.0, 1.0]), weights)
    assert not torch.allclose(combined, changed)


def test_weighted_group_gae_is_the_gae_of_weighted_total_reward():
    rng = np.random.default_rng(90)
    rewards, values, following = (rng.normal(size=(20, 3)) for _ in range(3))
    weights = np.array([1.0, 0.01, 0.1])
    ended = np.zeros(20, bool)
    ended[[5, 12, 19]] = True
    terminated = ended.copy()
    terminated[12] = False  # timeout bootstraps
    ticks = rng.integers(1, 4, size=20)
    group, returns = grouped_gae(rewards, values, following, terminated, ended, ticks)
    collapsed = []
    for array in (rewards, values, following):
        summed = np.zeros_like(array)
        summed[:, 0] = array @ weights
        collapsed.append(summed)
    total, total_returns = grouped_gae(*collapsed, terminated, ended, ticks)
    np.testing.assert_allclose(group @ weights, total[:, 0], atol=1e-6)
    np.testing.assert_allclose(returns @ weights, total_returns[:, 0], atol=1e-6)


@pytest.mark.parametrize("normalization", ["shared", "per_head"])
def test_constant_heads_are_finite_and_invalid_normalization_is_rejected(normalization):
    combined, columns = combine_advantages(torch.ones(8, 3), normalization=normalization)
    assert torch.count_nonzero(combined) == torch.count_nonzero(columns) == 0
    with pytest.raises(ValueError, match="normalization"):
        PPOConfig(advantage_normalization="unknown")


class ToyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(1))

    def forward(self, observation):
        return self.mean.expand(len(observation["x"]), 1)

    def action_mask(self, observation):
        return torch.ones_like(observation["x"])


class ToyPolicy(nn.Module):
    """One-step measured tracking reward; exact optimum .25/(1+.001)."""

    def __init__(self):
        super().__init__()
        self.actor = ToyActor()
        self.log_std = nn.Parameter(torch.tensor([-2.3]))
        self.critics = nn.ModuleList([nn.Linear(1, 1) for _ in range(3)])
        for critic in self.critics:
            nn.init.zeros_(critic.weight)
            nn.init.zeros_(critic.bias)

    def evaluate(self, observation, actions):
        distribution = torch.distributions.Normal(self.actor(observation), self.log_std.exp())
        values = torch.cat([critic(observation["x"]) for critic in self.critics], dim=-1)
        return distribution.log_prob(actions).sum(-1), distribution.entropy().sum(-1), values


def test_shared_objective_ppo_learns_tracking_with_small_motion_cost():
    torch.manual_seed(90)
    policy = ToyPolicy()
    learner = MultiCriticPPO(
        policy,
        PPOConfig(
            learning_rate=0.01, critic_learning_rate=0.01, epochs=4, batch_size=64, entropy_weight=0.0, target_kl=0.1
        ),
    )
    obs = {"x": torch.ones(128, 1)}
    initial_error = (policy.actor.mean.item() - 0.25) ** 2
    for _ in range(16):
        with torch.no_grad():
            action = torch.distributions.Normal(policy.actor(obs), policy.log_std.exp()).sample()
            log, _, value = policy.evaluate(obs, action)
        a = action[:, 0].numpy()
        rollout = {
            "observations": {"x": obs["x"].numpy()},
            "actions": action.numpy(),
            "log_prob": log.numpy(),
            "values": value.numpy(),
            "next_values": np.zeros((128, 3)),
            "rewards": np.c_[-((a - 0.25) ** 2), -0.001 * a * a, np.zeros(128)],
            "terminated": np.ones(128, bool),
            "episode_end": np.ones(128, bool),
            "ticks": np.ones(128),
        }
        learner.update(rollout)
    final_error = (policy.actor.mean.item() - 0.25) ** 2
    assert final_error < initial_error * 0.05
    assert abs(policy.actor.mean.item() - 0.25 / (1 + 0.001)) < 0.05


def test_actor_gradient_diagnostics_do_not_mutate_rng_parameters_or_optimizer_update():
    import copy

    torch.manual_seed(90)
    first = ToyPolicy()
    second = copy.deepcopy(first)
    cfg = PPOConfig(epochs=1, batch_size=8)
    measured, unmeasured = MultiCriticPPO(first, cfg), MultiCriticPPO(second, cfg)
    obs = {"x": torch.ones(8, 1)}
    with torch.no_grad():
        actions = torch.distributions.Normal(first.actor(obs), first.log_std.exp()).sample()
        old, _, value = first.evaluate(obs, actions)
    log, _, _ = first.evaluate(obs, actions)
    components = torch.randn(8, 3)
    rng = torch.get_rng_state().clone()
    stats = measured.actor_gradient_diagnostics((log - old).exp(), components)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None for p in first.parameters())
    assert np.isfinite(stats["actor_gradient_norm_by_head"]).all()
    rollout = {
        "observations": {"x": obs["x"].numpy()},
        "actions": actions.numpy(),
        "log_prob": old.numpy(),
        "values": value.numpy(),
        "next_values": np.zeros((8, 3)),
        "rewards": np.arange(24).reshape(8, 3) / 24,
        "terminated": np.ones(8, bool),
        "episode_end": np.ones(8, bool),
        "ticks": np.ones(8),
    }
    unmeasured.actor_gradient_diagnostics = lambda *args: {"disabled": True}
    torch.manual_seed(42)
    measured.update(rollout)
    torch.manual_seed(42)
    unmeasured.update(rollout)
    for name, tensor in first.state_dict().items():
        torch.testing.assert_close(tensor, second.state_dict()[name])
