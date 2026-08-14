# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from emet.controller.controller_dynamem import DynamemController


class _PlannerFail:
    def plan(self, start, goal):
        return SimpleNamespace(success=False, reason="blocked", trajectory=[])

    def clean_path_for_xy(self, waypoints):
        return waypoints


@pytest.fixture
def nav_agent(monkeypatch):
    robot = MagicMock()
    robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    params = {"eqa": {"habitat_perfect_nav": False}}
    agent = DynamemController.__new__(DynamemController)
    agent.robot = robot
    agent.parameters = params
    agent.rerun_visualizer = MagicMock()
    agent.pos_err_threshold = 0.1
    agent.rot_err_threshold = 0.1
    agent.space = MagicMock()
    agent.space.sample_navigation.return_value = np.array([1.0, 1.0, 0.0])
    agent.planner = _PlannerFail()
    agent._last_nav_attempt = None
    agent._episode_diagnostics_recorder = None
    monkeypatch.setattr(
        "emet.controller.controller_dynamem.habitat_perfect_nav_enabled",
        lambda _p: False,
    )
    monkeypatch.setattr(
        "emet.controller.controller_dynamem.is_habitat_robot_client",
        lambda _r: False,
    )
    return agent


def test_navigate_to_target_pose_returns_false_on_planner_failure(nav_agent):
    finished = nav_agent.navigate_to_target_pose(
        np.array([2.0, 2.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
    )
    assert not bool(finished)
    assert nav_agent._last_nav_attempt is not None
    assert nav_agent._last_nav_attempt.finished is False
    assert nav_agent._last_nav_attempt.note == "blocked"
    nav_agent.robot.execute_trajectory.assert_not_called()


def test_navigate_to_target_pose_uses_navmesh_when_enabled(monkeypatch):
    robot = MagicMock()
    robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    params = {"eqa": {"habitat_perfect_nav": True}}
    agent = DynamemController.__new__(DynamemController)
    agent.robot = robot
    agent.parameters = params
    agent.rerun_visualizer = MagicMock()
    agent._last_nav_attempt = None
    agent._episode_diagnostics_recorder = None
    nav_res = SimpleNamespace(
        success=True,
        finished=True,
        dist_m=0.5,
        method="habitat_navmesh",
        note="ok",
        target_obs_id=None,
    )
    monkeypatch.setattr(
        "emet.controller.controller_dynamem.habitat_perfect_nav_enabled",
        lambda _p: True,
    )
    monkeypatch.setattr(
        "emet.controller.controller_dynamem.is_habitat_robot_client",
        lambda _r: True,
    )
    monkeypatch.setattr(
        "emet.controller.controller_dynamem.habitat_navmesh_navigate",
        lambda *a, **k: nav_res,
    )
    finished = agent.navigate_to_target_pose(np.array([1.0, 2.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    assert bool(finished)
    assert agent._last_nav_attempt.method == "habitat_navmesh"
