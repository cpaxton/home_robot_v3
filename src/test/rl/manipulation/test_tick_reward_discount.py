# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Bundling executor-owned ticks must preserve the physical reward clock."""

import numpy as np
import pytest

from emet.rl.manipulation.multi_critic_ppo import MultiCriticPPO, PPOConfig, discount_tick_rewards, grouped_gae
from emet.rl.manipulation.whole_body_actor import MultiCriticPolicy
from emet.rl.manipulation.whole_body_training import collect_rollout

from .test_whole_body_v2 import observation


@pytest.mark.parametrize("gamma", [0.5, 0.9975, 1.0])
@pytest.mark.parametrize("terminated", [False, True])
@pytest.mark.parametrize("bonus", [2.0, -10.0])
def test_bundled_returns_match_tick_returns_including_terminal_bonus_and_timeout(gamma, terminated, bonus):
    rng = np.random.default_rng(90)
    ticks = np.array([1, 2, 49])
    starts = np.r_[0, ticks.cumsum()[:-1]]
    rewards = rng.normal(size=(ticks.sum(), 3))
    rewards[-1, 0] += bonus
    values = rng.normal(size=(len(rewards) + 1, 3))
    ended = np.arange(len(rewards)) == len(rewards) - 1
    _, expanded = grouped_gae(
        rewards, values[:-1], values[1:], ended & terminated, ended, np.ones(len(rewards)), gamma=gamma, lam=1.0
    )
    bundled = np.stack(
        [
            discount_tick_rewards(rewards[start : start + count], gamma=gamma)
            for start, count in zip(starts, ticks, strict=True)
        ]
    )
    _, actual = grouped_gae(
        bundled,
        values[starts],
        values[starts + ticks],
        [False, False, terminated],
        [False, False, True],
        ticks,
        gamma=gamma,
        lam=1.0,
    )
    np.testing.assert_allclose(actual, expanded[starts], rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("gamma", [0.5, 0.9975, 1.0])
def test_single_tick_reward_is_unchanged(gamma):
    rewards = np.array([[0.25, -0.0, -10.0]], dtype=np.float32)
    actual = discount_tick_rewards(rewards, gamma=gamma)
    assert actual.tobytes() == rewards[0].tobytes()


@pytest.mark.parametrize("rewards,gamma", [([], 0.5), ([[1, 2]], 0.5), ([[1, 2, np.nan]], 0.5), ([[1, 2, 3]], 0.0)])
def test_invalid_tick_rewards_are_rejected(rewards, gamma):
    with pytest.raises(ValueError, match="per-tick"):
        discount_tick_rewards(rewards, gamma=gamma)


class BundledEpisode:
    def __init__(self, *, ticks=49, include_ticks=True, inconsistent_total=False):
        self.ticks, self.include_ticks = ticks, include_ticks
        self.tick_rewards = np.tile(np.array([0.05, -0.01, -0.02], dtype=np.float32), (ticks, 1))
        self.tick_rewards[-1, 0] += 2.0
        self.info = {
            "elapsed_control_ticks": ticks,
            "reward_groups": self.tick_rewards.sum(0),
            "primitive_success": True,
        }
        if include_ticks:
            self.info["reward_group_ticks"] = self.tick_rewards.copy()
        if inconsistent_total:
            self.info["reward_groups"][0] += 2

    def reset(self):
        return {k: v[0].numpy() for k, v in observation().items()}, {}

    def step(self, action):
        return self.reset()[0], float(self.info["reward_groups"].sum()), True, False, self.info


def test_collector_uses_learner_gamma_and_keeps_undiscounted_episode_info():
    env = BundledEpisode()
    model = MultiCriticPolicy(hidden=16)
    callbacks = []
    raw = env.info["reward_groups"].copy()
    rollout, _ = collect_rollout(env, model, env.reset()[0], 2, gamma=0.5, episode_callback=callbacks.append)
    expected = sum(0.5**j * reward for j, reward in enumerate(env.tick_rewards))
    np.testing.assert_allclose(rollout["rewards"], np.tile(expected, (2, 1)))
    np.testing.assert_array_equal(env.info["reward_groups"], raw)
    assert len(callbacks) == 2 and callbacks[0]["primitive_success"] is True
    assert rollout["ticks"].tolist() == [49, 49] and rollout["reward_discount"] == 0.5
    # A collector configured with a different discount cannot silently feed PPO.
    with pytest.raises(ValueError, match="must match PPO gamma"):
        MultiCriticPPO(model).update(rollout)
    stats = MultiCriticPPO(model, PPOConfig(gamma=0.5, epochs=1, batch_size=2)).update(rollout)
    assert np.isfinite(stats["critic_losses"]).all()


@pytest.mark.parametrize("ticks,include_ticks,inconsistent_total", [(49, False, False), (49, True, True)])
def test_collector_rejects_missing_tick_timing_or_inconsistent_total(ticks, include_ticks, inconsistent_total):
    env = BundledEpisode(ticks=ticks, include_ticks=include_ticks, inconsistent_total=inconsistent_total)
    with pytest.raises(ValueError, match="tick rewards|individual tick"):
        collect_rollout(env, MultiCriticPolicy(hidden=16), env.reset()[0], 2)


def test_collector_keeps_existing_one_tick_general_environment_contract():
    env = BundledEpisode(ticks=1, include_ticks=False)
    rollout, _ = collect_rollout(env, MultiCriticPolicy(hidden=16), env.reset()[0], 2)
    np.testing.assert_array_equal(rollout["rewards"], np.tile(env.info["reward_groups"], (2, 1)))


@pytest.mark.sim
def test_native_pregrasp_preserves_hold_rewards_and_final_tick_bonus():
    from emet.rl.manipulation.tamp_primitive_env import TampPrimitiveEnv
    from emet.rl.manipulation.whole_body_training import load_development

    suite, _ = load_development()
    env = TampPrimitiveEnv(suite["train_cases"][:1], oracle=True)
    try:
        _, info = env.reset()
        while True:
            _, reward, done, truncated, info = env.step(info["oracle_action"])
            tick_rewards = info["reward_group_ticks"]
            assert tick_rewards.shape == (info["elapsed_control_ticks"], 3)
            np.testing.assert_allclose(tick_rewards.sum(0), info["reward_groups"], rtol=1e-5, atol=1e-6)
            assert reward == pytest.approx(float(info["reward_groups"].sum()))
            if done or truncated:
                break
        assert info["primitive_success"] and info["stop_confirmed"]
        assert info["elapsed_control_ticks"] > 1  # executor-owned terminal hold/stop ticks
        terms = info["reward_terms"]
        final_tick_tracking = 0.05 * (terms["hand_pose_score"] + terms["base_pose_score"])
        assert tick_rewards[-1, 0] == pytest.approx(final_tick_tracking + 2.0, abs=1e-6)
        discounted = discount_tick_rewards(tick_rewards, gamma=0.9975)
        assert discounted[0] < info["reward_groups"][0]
    finally:
        env.close()
