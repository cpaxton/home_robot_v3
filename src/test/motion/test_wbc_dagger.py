# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Subtask-suite and DAgger mechanics."""

from __future__ import annotations

import numpy as np
import pytest

from emet.rl.manipulation.wbc_dagger import (
    WbcSubtask,
    WbcTaskSuiteEnv,
    collect_dagger_data,
    default_subtasks,
    evaluate_suite,
)
from emet.rl.manipulation.wbc_training import WbcActorCritic


class _FakeSuite:
    names = ("arm", "base")
    observation_space = None
    action_space = None

    def __init__(self):
        self.active_name = "arm"
        self._step = 0

    def reset(self, *, seed=None, options=None):
        self.active_name = "arm" if not options else options["subtask"]
        self._step = 0
        return np.zeros(4, dtype=np.float32), {"subtask": self.active_name}

    def step(self, action):
        self._step += 1
        truncated = self._step >= 3
        return np.zeros(4, dtype=np.float32), 0.0, False, truncated, {"subtask": self.active_name}

    def teacher_action(self):
        return np.ones(2, dtype=np.float32)


def test_subtask_validation():
    with pytest.raises(ValueError, match="arm or base"):
        WbcSubtask("empty", False, False, False, 0.1, 1.0)
    with pytest.raises(ValueError, match="target_range"):
        WbcSubtask("arm", True, False, True, 0.0, 1.0)
    with pytest.raises(ValueError, match="base_goal_mode"):
        WbcSubtask("arm", True, False, True, 0.1, 1.0, "sideways")


def test_collect_dagger_labels_actor_visited_states():
    env = _FakeSuite()
    policy = WbcActorCritic(4, 2, hidden=8)
    observations, actions, counts = collect_dagger_data(env, policy, steps=7, seed=0, noise_std=0.0)
    assert observations.shape == (7, 4)
    assert actions.shape == (7, 2)
    assert np.allclose(actions, 1.0)
    assert sum(counts.values()) == 7


@pytest.mark.sim
def test_suite_teacher_and_separate_evaluation():
    env = WbcTaskSuiteEnv(default_subtasks(), seed=0)
    assert env.names == ("arm", "base", "combined")
    for name in env.names:
        observation, info = env.reset(seed=0, options={"subtask": name})
        assert info["subtask"] == name
        assert observation.shape == env.observation_space.shape
        assert env.teacher_action().shape == env.action_space.shape
    result = evaluate_suite(env, None, episodes_per_task=1, seed=0, teacher=True)
    assert set(result) == {*env.names, "aggregate"}
    assert result["aggregate"]["episodes"] == 3.0
    assert result["aggregate"]["success_rate"] == 1.0
    env.close()
