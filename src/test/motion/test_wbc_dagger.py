# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Subtask-suite and DAgger mechanics."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from emet.rl.manipulation.wbc_dagger import (
    WbcSubtask,
    WbcTaskSuiteEnv,
    collect_dagger_data,
    contact_subtasks,
    default_subtasks,
    evaluate_suite,
    subtasks_from_contract,
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


def test_contract_round_trip_preserves_subtask_semantics():
    from emet.rl.manipulation.wbc_dagger import subtasks_from_contract

    contract = {
        "schema_version": 1,
        "robot": "pandaomron",
        "control_dt": 0.05,
        "success_hold_s": 0.25,
        "collision_force_n": 10.0,
        "subtasks": [
            {
                "name": "arm",
                "arm_goal": True,
                "base_goal": False,
                "hold_base": True,
                "target_range": 0.15,
                "horizon_s": 5.0,
                "base_goal_mode": "forward",
                "weight": 0.25,
                "active_arms": None,
            }
        ],
    }
    tasks = subtasks_from_contract(contract)
    assert tasks == (WbcSubtask("arm", True, False, True, 0.15, 5.0, "forward", 0.25),)
    with pytest.raises(ValueError, match="subtasks"):
        subtasks_from_contract({"schema_version": 1, "subtasks": []})


def test_default_subtasks_accepts_robot_aliases():
    assert [task.name for task in default_subtasks("mars")] == [task.name for task in default_subtasks("innate_mars")]
    assert [task.name for task in default_subtasks("rb_y1")] == [task.name for task in default_subtasks("rby1")]
    assert [task.name for task in default_subtasks("panda_omron")] == [
        task.name for task in default_subtasks("pandaomron")
    ]


def test_contact_subtasks_include_payload_and_wide_posture():
    tasks = {task.name: task for task in contact_subtasks("pandaomron")}
    assert set(tasks) == {"arm", "base", "combined", "wide_arm", "carry"}
    assert tasks["wide_arm"].torso_range == (0.12, 0.32)
    assert tasks["carry"].payload_mass_range == (0.1, 1.2)
    assert tasks["carry"].gripper_close is True
    assert tasks["carry"].torso_range == (0.15, 0.32)
    with pytest.raises(ValueError, match="no contact curriculum"):
        contact_subtasks("innate_mars")


def test_subtasks_from_contract_preserves_payload_contract():
    tasks = contact_subtasks("pandaomron")
    contract = {"schema_version": 1, "subtasks": [asdict(task) for task in tasks]}
    assert subtasks_from_contract(contract) == tasks


def test_subtask_rejects_invalid_payload_range():
    with pytest.raises(ValueError, match="payload_mass_range"):
        WbcSubtask("carry", True, False, True, 0.2, 5.0, payload_mass_range=(1.0, 0.1))
    with pytest.raises(ValueError, match="torso_range"):
        WbcSubtask("wide", True, False, True, 0.2, 5.0, torso_range=(0.3, 0.1))
