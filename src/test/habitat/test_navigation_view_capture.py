# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from emet_habitat.robot_client import HabitatRobotClient


@pytest.mark.parametrize("enabled", [False, True])
def test_capture_occurs_at_actual_path_to_yaw_boundary(enabled):
    robot = HabitatRobotClient.__new__(HabitatRobotClient)
    robot.retain_navigation_views = enabled
    robot.get_observation = Mock(return_value=SimpleNamespace(rgb="arrival"))
    robot._pending_nav = {"phase": "path", "wp_i": 0, "waypoints": [], "goal_theta": 1.2}
    assert robot.nav_tick(max_sim_steps=1) == "running"
    assert robot._pending_nav["phase"] == "yaw"
    assert robot.get_observation.call_count == int(enabled)
    if enabled:
        assert robot.pre_turn_observation.rgb == "arrival"


def test_observation_preserves_render_acquisition_timestamp(monkeypatch):
    from emet_habitat import robot_client

    robot = HabitatRobotClient.__new__(HabitatRobotClient)
    robot._expose_semantics = False
    robot._observation_kwargs = lambda: {}
    robot._sim = SimpleNamespace(
        get_frame=lambda: SimpleNamespace(
            rgb=None,
            depth=None,
            agent_state=None,
            intrinsics=None,
            acquisition_timestamp_ns=12345,
        )
    )
    monkeypatch.setattr(robot_client, "habitat_rgb_depth_to_observations", lambda **kwargs: SimpleNamespace())
    assert robot.get_observation().image_timing["head_cam_left"]["timestamp_ns"] == 12345


def test_navigation_step_retains_existing_observation_without_map_or_extra_step():
    robot = HabitatRobotClient.__new__(HabitatRobotClient)
    robot._sim = SimpleNamespace(step=Mock(return_value="rendered"))
    robot._sync_pose_from_sim = Mock()
    robot._post_step_hooks = []
    robot._pending_nav = {"phase": "path"}
    robot.navigation_view_history = SimpleNamespace(add=Mock())
    robot.get_observation = Mock(return_value="registered RGB-D")
    assert robot._sim_step("move_forward") == "rendered"
    robot._sim.step.assert_called_once_with("move_forward")
    robot.navigation_view_history.add.assert_called_once_with("registered RGB-D")
    robot._pending_nav = {"phase": "done"}
    robot._sim_step("turn_left")
    assert robot.navigation_view_history.add.call_count == 1
