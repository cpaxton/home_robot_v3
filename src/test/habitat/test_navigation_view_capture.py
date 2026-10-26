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
