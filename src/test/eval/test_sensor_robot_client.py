# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from types import SimpleNamespace

import numpy as np
import pytest

from emet.eval.grapheqa_navigation import predicted_camera_pose
from emet.eval.grapheqa_sensor import command_camera_pose
from emet.eval.sensor_robot_client import SensorRobotClient
from emet.robots.footprint import CircularFootprint


class Sensor:
    def __init__(self):
        self.actions = []
        self.last = {
            "camera_pose": command_camera_pose(yaw_deg=90, tilt_deg=-30, sensor_height=1.5),
            "rgb": np.zeros((8, 10, 3), np.uint8),
            "depth": np.ones((8, 10), np.float32),
            "camera_K": np.eye(3),
            "sequence": 1,
            "timestamp_ns": 12,
        }

    def capture(self):
        self.last = {**self.last, "sequence": self.last["sequence"] + 1}
        return self.last

    def step(self, action, amount):
        self.actions.append((action, amount))
        self.last["camera_pose"] = predicted_camera_pose(self.last["camera_pose"], action, amount)
        return self.capture()


def robot():
    nav = SimpleNamespace(footprint=CircularFootprint(0.1), update=lambda frame: None)
    return SensorRobotClient(Sensor(), nav)


def test_observation_uses_only_sensor_estimate_and_declares_acquisition():
    client = robot()
    obs = client.get_observation()
    np.testing.assert_allclose(obs.compass, [np.pi / 2])
    np.testing.assert_array_equal(obs.camera_pose, client.sensor.last["camera_pose"])
    assert obs.semantic is None and obs.emet_session is None
    assert obs.image_timing["head_cam_left"]["timestamp_ns"] == 12
    assert obs.seq_id == client._seq_id == 2
    assert not hasattr(client, "_sim") and not hasattr(client, "find_path_to_xy")
    assert client.get_footprint() is client.navigation.footprint


def test_tracking_loss_never_becomes_command_pose():
    client = robot()
    client.sensor.last["camera_pose"] = None
    with pytest.raises(RuntimeError, match="tracking lost"):
        client.get_base_pose()


def test_relative_motion_converts_using_measured_heading():
    client = robot()
    routes = []
    client.executor.execute = lambda route: routes.append(route) or {"success": False, "reason": "unknown_floor"}
    assert client.move_base_to([1, 0, 0], relative=True) is False
    np.testing.assert_allclose(routes[0][0], [0, 1, np.pi / 2], atol=1e-12)
    assert client.last_motion_failed() and not client.at_goal()


def test_impossible_gaze_rejects_without_motion_or_false_feedback():
    client = robot()
    before = client.get_pan_tilt()
    assert client.head_to(0.3, -0.5) is False
    assert client.head_to(0, 0.5) is False
    assert client.sensor.actions == []
    assert client.get_pan_tilt() == before
    assert not client.get_head_capability().contains([0.3, -0.5])


def test_supported_head_motion_uses_bounded_measured_steps():
    client = robot()
    assert client.head_to(0, np.deg2rad(-42))
    assert all(amount <= 5 for _, amount in client.sensor.actions)
    assert np.isclose(client.get_pan_tilt()[1], np.deg2rad(-42))
    assert client._seq_id > 1


def test_sensor_body_does_not_pretend_manipulation_is_supported():
    with pytest.raises(NotImplementedError, match="no manipulator"):
        robot().move_to_manip_posture()


def test_navigation_sampling_does_not_share_model_cpu_rng(monkeypatch):
    import torch

    from emet.eval.grapheqa_navigation import ObservedSpaceNavigation

    draws = []
    frame = Sensor().last
    for seed in (13, 42):
        nav = ObservedSpaceNavigation()
        monkeypatch.setattr(nav.map, "add", lambda **kwargs: draws.append(torch.rand(5)))
        monkeypatch.setattr(nav.map, "get_navigation_map", lambda: (torch.zeros((256, 256), dtype=torch.bool),) * 2)
        torch.manual_seed(seed)
        before = torch.get_rng_state().clone()
        nav.update(frame)
        assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(*draws)
