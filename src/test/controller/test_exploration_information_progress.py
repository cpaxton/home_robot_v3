# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.controller.dynamem.navigation import _execute_validated_waypoints, _record_arrival_coverage, run_exploration
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
    fields = agent._record_nav_plan_fields.call_args_list[0].kwargs
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


def test_next_segment_is_revalidated_after_measured_arrival(monkeypatch):
    monkeypatch.setattr("emet.controller.operations.payload_verification.verify_carried_object", lambda _: None)
    monkeypatch.setattr(
        "emet.controller.dynamem.look.wait_post_motion_obs", lambda robot, **_: setattr(robot, "_seq_id", 2)
    )
    observations = []
    agent = SimpleNamespace(
        space=SimpleNamespace(obstacle_map_mode="physical"),
        robot=SimpleNamespace(execute_trajectory=Mock(return_value=True), _seq_id=1),
        voxel_map=SimpleNamespace(observations=observations),
        planner=SimpleNamespace(reset=Mock()),
        update=Mock(side_effect=lambda **_: observations.append("new observation")),
        _current_planning_xyt=Mock(return_value=np.zeros(3)),
        _filter_unsafe_nav_traj=Mock(side_effect=[([[0, 0, 1]], None, 0.4), ([], "new_obstacle", None)]),
        _record_nav_plan_fields=Mock(),
        pos_err_threshold=0.07,
        rot_err_threshold=0.15,
    )
    assert _execute_validated_waypoints(agent, [[0, 0, 1], [1, 0, 1]], 10) == (False, "new_obstacle")
    agent.robot.execute_trajectory.assert_called_once()
    agent.update.assert_called_once_with(full_perception=False)
    assert agent.planner.reset.call_count == 2


@pytest.mark.parametrize(
    "fresh,reason", [(False, "stale_navigation_observation"), (True, "navigation_map_update_missing")]
)
def test_arrival_without_fresh_mapping_stops_before_next_segment(monkeypatch, fresh, reason):
    monkeypatch.setattr("emet.controller.operations.payload_verification.verify_carried_object", lambda _: None)
    monkeypatch.setattr(
        "emet.controller.dynamem.look.wait_post_motion_obs",
        lambda robot, **_: setattr(robot, "_seq_id", 2 if fresh else 1),
    )
    agent = SimpleNamespace(
        space=SimpleNamespace(obstacle_map_mode="physical"),
        robot=SimpleNamespace(execute_trajectory=Mock(return_value=True), _seq_id=1),
        voxel_map=SimpleNamespace(observations=[]),
        planner=SimpleNamespace(reset=Mock()),
        update=Mock(),
        _current_planning_xyt=lambda: np.zeros(3),
        _filter_unsafe_nav_traj=lambda *a, **k: ([], None, 0.4),
        pos_err_threshold=0.07,
        rot_err_threshold=0.15,
    )
    assert _execute_validated_waypoints(agent, [[0, 0, 1], [1, 0, 1]], 10) == (False, reason)
    agent.robot.execute_trajectory.assert_called_once()
    assert agent.update.call_count == int(fresh)


@pytest.mark.parametrize("source", ["da3", "lingbot"])
def test_physical_geometry_update_cannot_reproject_old_inferred_depth(source):
    from emet.controller.dynamem.perception import update

    class ReachedCurrentDepthResolver(Exception):
        pass

    obs = SimpleNamespace(rgb=np.zeros((2, 2, 3)), depth=None, camera_K=np.eye(3), camera_pose=np.eye(4))
    agent = SimpleNamespace(
        robot=SimpleNamespace(get_observation=lambda: obs),
        space=SimpleNamespace(obstacle_map_mode="physical"),
        obs_count=1,
        _run_full_perception=lambda _: False,
        _depth_source=source,
        _da3_infer_every_n=10,
        _lingbot_infer_every_n=10,
        _da3_last_depth=np.ones((2, 2)),
        _lingbot_last_depth=np.ones((2, 2)),
        _resolve_depth_map=Mock(side_effect=ReachedCurrentDepthResolver),
    )
    with pytest.raises(ReachedCurrentDepthResolver):
        update(agent, full_perception=False)
    agent._resolve_depth_map.assert_called_once()


def test_ambiguous_execution_cannot_continue_to_later_waypoints():
    agent = SimpleNamespace(
        robot=SimpleNamespace(execute_trajectory=Mock(return_value=None)),
        pos_err_threshold=0.07,
        rot_err_threshold=0.15,
    )
    assert _execute_validated_waypoints(agent, [[0, 0, 1]], 10) == (False, "ambiguous_navigation_result")
