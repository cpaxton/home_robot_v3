# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.controller.dynamem.navigation import _record_arrival_coverage, run_exploration
from emet.core.navigation_result import NavigationRoute


@pytest.mark.parametrize(
    "distance,gain,status,expected",
    [
        (0, True, True, True),
        (1, False, True, False),
        (0, True, None, False),
        (0.5, True, False, True),
    ],
)
def test_physical_exploration_separates_motion_from_sensor_gain(distance, gain, status, expected):
    before = np.zeros((2, 2), dtype=bool)
    after = before.copy()
    after[0, 0] = gain
    agent = SimpleNamespace(
        space=SimpleNamespace(obstacle_map_mode="physical"),
        voxel_map=SimpleNamespace(grid_resolution=0.1, get_sensor_observed_cells=Mock(side_effect=[before, after])),
        _current_planning_xyt=Mock(side_effect=[np.zeros(3), np.array([distance, 0, 1])]),
        execute_action=Mock(return_value=(status, np.ones(3))),
        _last_nav_plan={"new_sensor_cells": int(gain)},
        announce_action=Mock(),
        _record_nav_plan_fields=Mock(),
        _mark_nav_goal_blocked=Mock(),
        _maybe_emit_navgrid_ascii=Mock(),
    )
    assert run_exploration(agent) is expected
    fields = agent._record_nav_plan_fields.call_args.kwargs
    assert fields["measured_distance_m"] == distance
    assert fields["motion_outcome"] == ("failed" if status is None else "reached" if status else "partial")


def test_coverage_uses_pre_motion_snapshot_not_startup_growth():
    before = np.ones((2, 2), dtype=bool)
    agent = SimpleNamespace(
        space=SimpleNamespace(obstacle_map_mode="physical"),
        voxel_map=SimpleNamespace(grid_resolution=0.1, get_sensor_observed_cells=lambda: before.copy()),
        _record_nav_plan_fields=Mock(),
    )
    _record_arrival_coverage(agent, before)
    assert agent._record_nav_plan_fields.call_args.kwargs["new_sensor_cells"] == 0


def test_legacy_route_boundary_keeps_semantic_target_out_of_motion():
    target = [1, 2, 3]
    route = NavigationRoute.from_value([[0, 0, 0], [np.nan] * 3, target])
    assert route.waypoints == [[0, 0, 0]]
    assert route.target_xyz == target and route.finished
    assert NavigationRoute.from_value(route) is route
    assert not NavigationRoute.from_value([])
