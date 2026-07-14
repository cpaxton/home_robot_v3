# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the root directory).

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from emet.core.zmq_protocol import EMET_ZMQ_ROBOT_ID_KEY, EMET_ZMQ_SESSION_KEY

try:
    from emet_habitat.zmq_server import HabitatZmqServer
except ImportError:
    pytest.skip("emet_habitat not installed (use .venv-habitat)", allow_module_level=True)


def _mock_habitat_robot() -> MagicMock:
    robot = MagicMock()
    robot.get_base_pose.return_value = np.array([1.0, 2.0, 0.5], dtype=np.float64)
    robot.get_pan_tilt.return_value = (0.0, np.deg2rad(-30.0))
    # Realistic pinhole K for 48x64 (cx≈31.5, cy≈23.5).
    camera_k = np.array([[100.0, 0.0, 31.5], [0.0, 100.0, 23.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    robot.get_observation.return_value = MagicMock(
        rgb=np.zeros((48, 64, 3), dtype=np.uint8),
        depth=np.ones((48, 64), dtype=np.float32) * 1.5,
        camera_K=camera_k,
        camera_pose=np.eye(4),
    )
    return robot


def test_habitat_zmq_scales_camera_k_with_image():
    robot = _mock_habitat_robot()
    full_k = np.asarray(robot.get_observation().camera_K, dtype=np.float64)
    server = HabitatZmqServer(robot, scene_id="test-scene", port_offset=9996, image_scaling=0.5)
    with (
        patch("emet_habitat.zmq_server.compression.to_jpg", side_effect=lambda x: x),
        patch("emet_habitat.zmq_server.compression.to_jp2", side_effect=lambda x: x),
    ):
        msg = server.get_full_observation_message()
    assert msg is not None
    assert msg["rgb_height"] == 24
    assert msg["rgb_width"] == 32
    k = np.asarray(msg["camera_K"], dtype=np.float64)
    np.testing.assert_allclose(k[0, 0], full_k[0, 0] * 0.5, atol=1e-6)
    np.testing.assert_allclose(k[1, 1], full_k[1, 1] * 0.5, atol=1e-6)
    np.testing.assert_allclose(k[0, 2], full_k[0, 2] * 0.5, atol=1e-6)
    np.testing.assert_allclose(k[1, 2], full_k[1, 2] * 0.5, atol=1e-6)


def test_habitat_zmq_full_obs_publishes_stretch_contract():
    robot = _mock_habitat_robot()
    server = HabitatZmqServer(robot, scene_id="test-scene", port_offset=9999)
    with (
        patch("emet_habitat.zmq_server.compression.to_jpg", side_effect=lambda x: x),
        patch("emet_habitat.zmq_server.compression.to_jp2", side_effect=lambda x: x),
    ):
        msg = server.get_full_observation_message()
    assert msg is not None
    assert msg[EMET_ZMQ_ROBOT_ID_KEY] == "stretch"
    assert EMET_ZMQ_SESSION_KEY in msg
    assert msg[EMET_ZMQ_SESSION_KEY]["navigation_origin_xyt"] == [1.0, 2.0, 0.5]
    np.testing.assert_allclose(msg["gps"], [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(msg["compass"], [0.0], atol=1e-6)
    assert msg["is_simulation"] is True


def test_habitat_zmq_servo_message_stretch_contract():
    robot = _mock_habitat_robot()
    server = HabitatZmqServer(robot, scene_id="test-scene", port_offset=9997)
    with (
        patch("emet_habitat.zmq_server.compression.to_jpg", side_effect=lambda x: x),
        patch("emet_habitat.zmq_server.compression.to_jp2", side_effect=lambda x: x),
    ):
        msg = server.get_servo_message()
    assert msg is not None
    assert "head_color_image" in msg
    assert "head_depth_image" in msg
    assert "joint_positions" in msg
    assert "base_pose" in msg
    assert msg[EMET_ZMQ_ROBOT_ID_KEY] == "stretch"


def test_habitat_zmq_handle_action_nav_world():
    robot = _mock_habitat_robot()
    server = HabitatZmqServer(robot, scene_id="test-scene", port_offset=9998)
    server.handle_action({"xyt": [3.0, 4.0, 0.1], "nav_world": True, "nav_relative": False})
    robot.move_base_to.assert_called_once()
    args, kwargs = robot.move_base_to.call_args
    np.testing.assert_allclose(args[0], [3.0, 4.0, 0.1], atol=1e-6)
    assert kwargs.get("relative") is False


def test_habitat_zmq_handle_action_head_to():
    robot = _mock_habitat_robot()
    robot.get_pan_tilt.return_value = (0.4, -0.5)
    server = HabitatZmqServer(robot, scene_id="test-scene", port_offset=9995)
    server.handle_action({"head_to": [0.4, -0.5], "step": 1})
    robot.head_to.assert_called_once()
    args, kwargs = robot.head_to.call_args
    assert args[0] == pytest.approx(0.4)
    assert args[1] == pytest.approx(-0.5)
    assert server._at_goal is True
    state = server.get_state_message()
    assert state is not None
    from emet.motion.kinematics import HelloStretchIdx

    q = np.asarray(state["joint_positions"], dtype=np.float64)
    assert q[HelloStretchIdx.HEAD_PAN] == pytest.approx(0.4)
    assert q[HelloStretchIdx.HEAD_TILT] == pytest.approx(-0.5)


def test_habitat_zmq_handle_action_joint_ack_without_xyt():
    robot = _mock_habitat_robot()
    server = HabitatZmqServer(robot, scene_id="test-scene", port_offset=9994)
    server.handle_action({"joint": [0.0] * 6, "step": 2})
    robot.move_base_to.assert_not_called()
    assert server._at_goal is True
