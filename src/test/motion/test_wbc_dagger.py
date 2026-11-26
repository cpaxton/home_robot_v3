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


def test_dual_arm_robots_default_to_single_arm_subtasks():
    tasks = {task.name: task for task in default_subtasks("sourccey")}
    assert set(tasks) == {"arm_left", "arm_right", "base", "combined_left", "combined_right"}
    assert tasks["arm_left"].active_arms == ("left",)
    assert tasks["combined_right"].active_arms == ("right",)
    assert tasks["arm_left"].arm_goal and not tasks["arm_left"].base_goal
    assert tasks["combined_left"].arm_goal and tasks["combined_left"].base_goal
    with pytest.raises(ValueError, match="active_arms"):
        WbcSubtask("arm", True, False, True, 0.1, 1.0, active_arms=())


def test_cap_dataset_bounds_size_and_keeps_pairs():
    from emet.rl.manipulation.wbc_dagger import _cap_dataset

    index = np.arange(50)
    observations = np.column_stack([index, index])
    actions = index.astype(np.float64)
    capped_observations, capped_actions = _cap_dataset(observations, actions, 20, seed=0)
    assert capped_observations.shape == (20, 2)
    assert capped_actions.shape == (20,)
    assert np.array_equal(capped_observations[:, 0], capped_actions)


def test_selection_score_balances_the_weakest_subtask():
    from emet.rl.manipulation.wbc_dagger import _selection_score

    evaluation = {
        "aggregate": {"success_rate": 0.84},
        "left": {"success_rate": 0.9},
        "right": {"success_rate": 0.5},
    }
    assert _selection_score(evaluation, "aggregate") == pytest.approx(0.84)
    assert _selection_score(evaluation, "balanced") == pytest.approx(0.84 + 0.25 * 0.5)
    with pytest.raises(ValueError, match="unknown checkpoint selection"):
        _selection_score(evaluation, "min")


def test_default_subtask_windows_and_collection_weights():
    mars = {task.name: task for task in default_subtasks("innate_mars")}
    assert mars["arm"].horizon_s == 3.0
    assert mars["combined"].horizon_s == 5.0
    stretch = {task.name: task for task in default_subtasks("stretch")}
    assert stretch["arm"].horizon_s == 5.0
    assert stretch["combined"].horizon_s == 8.0
    assert stretch["combined"].weight == pytest.approx(0.6)
    sourccey = {task.name: task for task in default_subtasks("sourccey")}
    assert sourccey["base"].weight == pytest.approx(0.1)
    assert sourccey["combined_left"].weight == pytest.approx(0.3)
    assert sourccey["combined_right"].weight == pytest.approx(0.3)


def test_default_subtasks_accepts_robot_aliases():
    assert [task.name for task in default_subtasks("mars")] == [task.name for task in default_subtasks("innate_mars")]
    assert [task.name for task in default_subtasks("rb_y1")] == [task.name for task in default_subtasks("rby1")]
    assert [task.name for task in default_subtasks("panda_omron")] == [
        task.name for task in default_subtasks("pandaomron")
    ]
