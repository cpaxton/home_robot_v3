# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Waypoint-timeout abort marks frontier goals blocked for explore replan."""

from __future__ import annotations

import numpy as np

from emet.controller.habitat_nav import goal_key_xy


class _TightStartPlanner:
    _clearance_m = np.ones((3, 3), dtype=np.float64)

    @staticmethod
    def is_explored_xy(_xy):
        return True

    @staticmethod
    def clearance_at_xy(xy):
        return 0.15 if float(xy[0]) == 0.0 else 0.30

    @staticmethod
    def to_pt(xy):
        return int(round(float(xy[0]) * 10)), int(round(float(xy[1]) * 10))

    @staticmethod
    def is_in_line_of_sight(_start, _end):
        return False


def test_nav_filter_allows_first_segment_to_escape_tight_start():
    from emet.controller.controller_dynamem import DynamemController

    agent = DynamemController.__new__(DynamemController)
    agent.planner = _TightStartPlanner()
    agent._min_clearance_m = 0.22
    start = [0.0, 0.0, 0.0]
    escape = [0.3, 0.0, 0.0]

    filtered, reason, min_clearance = agent._filter_unsafe_nav_traj([start, escape], start_xyt=start)

    assert reason is None
    assert filtered == [start, escape]
    assert min_clearance == 0.15


def test_nav_filter_still_rejects_later_unsafe_segment():
    from emet.controller.controller_dynamem import DynamemController

    agent = DynamemController.__new__(DynamemController)
    agent.planner = _TightStartPlanner()
    agent._min_clearance_m = 0.22
    start = [0.0, 0.0, 0.0]
    escape = [0.3, 0.0, 0.0]
    later = [0.6, 0.0, 0.0]

    filtered, reason, _ = agent._filter_unsafe_nav_traj([start, escape, later], start_xyt=start)

    assert filtered == [start, escape]
    assert reason is None


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
