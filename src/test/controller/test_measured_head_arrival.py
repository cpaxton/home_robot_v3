# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Head waits must distinguish arrival, off-target settling and missing telemetry."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.controller import zmq_client
from emet.controller.zmq_client import StretchZmqClient
from emet.motion.kinematics import HelloStretchIdx


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(zmq_client.timeit, "default_timer", lambda: now[0])
    monkeypatch.setattr(zmq_client.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    return now


def client(pose, velocity=None):
    return SimpleNamespace(
        _finish=False,
        _head_pan_tolerance=0.01,
        _head_tilt_tolerance=0.01,
        _head_not_moving_tolerance=0.0001,
        _scaled_motion_timeout=lambda t: t,
        out_of_date=lambda: False,
        get_joint_state=lambda: (pose.copy(), np.zeros(11) if velocity is None else velocity.copy(), None),
        send_message=Mock(),
    )


def test_stationary_off_target_is_failure(clock):
    robot = client(np.zeros(11))
    target = np.zeros(11)
    target[HelloStretchIdx.HEAD_TILT] = -0.5
    assert not StretchZmqClient._wait_for_head(robot, target, timeout=0.8, resend_action={"head_to": [0, -0.5]})
    assert clock[0] >= 0.8
    assert robot.send_message.call_count >= 1


def test_arrival_requires_tilt_to_stop(clock):
    velocity = np.zeros(11)
    velocity[HelloStretchIdx.HEAD_TILT] = 0.1
    robot = client(np.zeros(11), velocity)
    assert not StretchZmqClient._wait_for_head(robot, np.zeros(11), timeout=0.8)


def test_arrived_and_settled_succeeds(clock):
    assert StretchZmqClient._wait_for_head(client(np.zeros(11)), np.zeros(11), timeout=0.8)
    assert 0.35 < clock[0] < 0.8


@pytest.mark.parametrize("missing", ["stale", "joints"])
def test_missing_telemetry_is_bounded(clock, missing):
    robot = client(np.zeros(11))
    if missing == "stale":
        robot.out_of_date = lambda: True
    else:
        robot.get_joint_state = lambda: (None, None, None)
    assert not StretchZmqClient._wait_for_head(robot, np.zeros(11), timeout=0.8)
    assert clock[0] >= 0.8


@pytest.mark.parametrize("arrived", [True, False])
def test_head_and_look_propagate_arrival(clock, arrived):
    robot = SimpleNamespace(
        get_head_capability=lambda: SimpleNamespace(contains=lambda _: True, pan=(-3.0, 3.0), tilt=(-1.5, 0.0)),
        send_action=Mock(return_value={"step": 1}),
        _robot_model=SimpleNamespace(dof=11),
        _wait_for_head=Mock(return_value=arrived),
    )
    assert StretchZmqClient.head_to(robot, 0.0, -0.5, blocking=True) is arrived
    robot.head_to = Mock(return_value=arrived)
    assert StretchZmqClient.look_front(robot) is arrived
