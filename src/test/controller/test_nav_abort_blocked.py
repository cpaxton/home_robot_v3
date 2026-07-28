# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Waypoint-timeout abort marks frontier goals blocked for explore replan."""

from __future__ import annotations

from emet.controller.habitat_nav import goal_key_xy


def test_mark_nav_goal_blocked_from_last_plan():
    from emet.controller.controller_dynamem import DynamemController

    agent = DynamemController.__new__(DynamemController)
    agent._habitat_blocked_goals = set()
    agent._habitat_recent_goals = []
    agent._last_nav_plan = {
        "goal_xyt": [1.25, 3.5, 0.0],
        "traj": [[0.0, 0.0, 0.0], [1.25, 3.5, 0.1]],
    }
    agent._mark_nav_goal_blocked(reason="aborted_waypoint_timeout")
    key = goal_key_xy((1.25, 3.5))
    assert key in agent._habitat_blocked_goals
    assert key in agent._habitat_recent_goals
    assert agent._last_nav_plan.get("blocked_after_abort") is True
    assert agent._last_nav_plan.get("outcome") == "aborted_waypoint_timeout"


def test_mark_nav_goal_blocked_habitat_navmesh_stuck_reason():
    from emet.controller.controller_dynamem import DynamemController

    agent = DynamemController.__new__(DynamemController)
    agent._habitat_blocked_goals = set()
    agent._habitat_recent_goals = []
    agent._last_nav_plan = {
        "goal_xyt": [2.0, 4.0, 0.0],
        "method": "habitat_navmesh",
    }
    agent._mark_nav_goal_blocked(reason="habitat_navmesh_already_at_goal")
    key = goal_key_xy((2.0, 4.0))
    assert key in agent._habitat_blocked_goals
    assert agent._last_nav_plan.get("outcome") == "habitat_navmesh_already_at_goal"
