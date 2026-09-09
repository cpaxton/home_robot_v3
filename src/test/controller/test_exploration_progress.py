# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.controller.dynamem.navigation import run_exploration
from emet.controller.dynamem.pose import _best_frontier_point_from_graph
from emet.controller.habitat_nav import goal_key_xy


@pytest.mark.parametrize(
    "status,distance,expected", [(True, 0.0, False), (True, 0.5, True), (False, 0.5, True), (None, 0.0, False)]
)
def test_exploration_reports_progress_not_just_a_plan(status, distance, expected):
    agent = SimpleNamespace(
        announce_action=Mock(),
        execute_action=Mock(return_value=(status, [1, 2, 0])),
        _current_planning_xyt=Mock(side_effect=[[0, 0, 0], [distance, 0, 0]]),
        _mark_nav_goal_blocked=Mock(),
        _maybe_emit_navgrid_ascii=Mock(),
    )
    assert run_exploration(agent) is expected
    assert agent._last_nav_attempt.dist_m == distance
    assert agent._last_nav_attempt.goal_xy == (1, 2)
    assert agent._last_nav_attempt.finished is (bool(status) and expected)
    assert agent._mark_nav_goal_blocked.called is (status is not None and not expected)


@pytest.mark.parametrize("query", [None, "kitchen"])
def test_graph_frontier_fallback_honors_blocked_goals(query):
    near = SimpleNamespace(is_frontier=True, xyz=np.array([1, 0, 1]), labels=["kitchen"])
    far = SimpleNamespace(is_frontier=True, xyz=np.array([2, 0, 1]), labels=["kitchen"])
    agent = SimpleNamespace(
        graph_memory=SimpleNamespace(get_nodes=lambda: [near, far]),
        _habitat_blocked_goals={goal_key_xy(near.xyz[:2])},
        robot=None,
    )
    np.testing.assert_allclose(_best_frontier_point_from_graph(agent, query), far.xyz)


def test_ranked_graph_frontier_also_honors_blocked_goals():
    from emet.controller.controller_graph_eqa import GraphEQAController

    near = SimpleNamespace(source="frontier", xyz=np.array([1, 0, 1]))
    far = SimpleNamespace(source="frontier", xyz=np.array([2, 0, 1]))
    agent = GraphEQAController.__new__(GraphEQAController)
    agent.graph_memory = SimpleNamespace(hypothesize_nav_targets=lambda *a, **k: [near, far])
    agent.robot = SimpleNamespace(get_base_pose=lambda: np.zeros(3))
    agent._planning_base_xyt = lambda pose: pose
    agent._habitat_blocked_goals = {goal_key_xy(near.xyz[:2])}
    np.testing.assert_allclose(agent._best_frontier_point_from_graph("bowl"), far.xyz)


def test_blocked_goal_invalidates_saved_trajectory_continuation():
    from emet.controller.dynamem.navigation import _mark_nav_goal_blocked

    agent = SimpleNamespace(
        space=SimpleNamespace(traj=[[1, 0, 0], [np.nan] * 3, [2, 0, 0]]),
        _last_nav_plan={"goal_xyt": [1, 0, 0], "object_xyz": [2, 0, 0]},
        _record_nav_plan_fields=Mock(),
    )
    _mark_nav_goal_blocked(agent, reason="aborted_waypoint_timeout")
    assert agent.space.traj is None
    assert goal_key_xy((1, 0)) in agent._habitat_blocked_goals
    assert goal_key_xy((2, 0)) in agent._habitat_blocked_goals
