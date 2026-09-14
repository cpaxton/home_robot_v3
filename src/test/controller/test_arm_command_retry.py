# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from itertools import chain, count, repeat
from unittest.mock import Mock

import numpy as np
import pytest

from emet.controller.zmq_client import StretchZmqClient
from emet.motion import HelloStretchIdx

from . import make_zmq_test_client


def test_arm_retry_reuses_command_identity(monkeypatch):
    client = make_zmq_test_client(StretchZmqClient)
    client._finish = False
    client.in_manipulation_mode = Mock(return_value=True)
    for joint in ("arm", "lift", "base_x", "wrist_roll", "wrist_pitch", "wrist_yaw"):
        setattr(client, f"_{joint}_joint_tolerance", 0.01)
    sent = {"joint": [0] * 6, "command": {"sequence": 9}, "step": 12}
    client.send_action = Mock(return_value=sent)
    client.send_message = Mock()
    states = chain([(np.ones(11), np.ones(11), None)] * 42, repeat((np.zeros(11), np.zeros(11), None)))
    client.get_joint_state = Mock(side_effect=states)
    monkeypatch.setattr("emet.controller.zmq_client.time.sleep", lambda _: None)
    monkeypatch.setattr("emet.controller.zmq_client.timeit.default_timer", lambda: next(clock))
    clock = count(0, 0.01)
    assert client.arm_to([0] * 6, blocking=True) is True
    client.send_action.assert_called_once()
    client.send_message.assert_called_once_with(sent)


@pytest.mark.parametrize("axis", [HelloStretchIdx.BASE_X, HelloStretchIdx.BASE_THETA, HelloStretchIdx.ARM])
def test_arm_completion_requires_stopped_motion_not_only_position(axis, monkeypatch):
    client = make_zmq_test_client(StretchZmqClient)
    client._finish = False
    client.in_manipulation_mode = Mock(return_value=True)
    for joint in ("arm", "lift", "base_x", "wrist_roll", "wrist_pitch", "wrist_yaw"):
        setattr(client, f"_{joint}_joint_tolerance", 0.01)
    client.send_action = Mock(return_value={"joint": [0] * 6})
    client.send_message = Mock()
    velocity = np.zeros(11)
    velocity[axis] = 0.2
    samples = chain([(np.zeros(11), velocity, None)] * 10, repeat((np.zeros(11), np.zeros(11), None)))
    client.get_joint_state = Mock(side_effect=samples)
    clock = count(0, 0.02)
    monkeypatch.setattr("emet.controller.zmq_client.timeit.default_timer", lambda: next(clock))
    monkeypatch.setattr("emet.controller.zmq_client.time.sleep", lambda _: None)
    assert client.arm_to([0] * 6)
    assert client.get_joint_state.call_count >= 16


@pytest.mark.parametrize("ratio, expected", [(None, False), (1.0, False), (2.0, False), (0.5, True)])
def test_arm_wait_uses_existing_sim_time_scale_without_changing_hardware_budget(ratio, expected, monkeypatch):
    client = make_zmq_test_client(StretchZmqClient)
    client._state = {} if ratio is None else {"sim_to_real_ratio": ratio}
    client.in_manipulation_mode = Mock(return_value=True)
    for joint in ("arm", "lift", "base_x", "wrist_roll", "wrist_pitch", "wrist_yaw"):
        setattr(client, f"_{joint}_joint_tolerance", 0.01)
    client.send_action = Mock(return_value={"joint": [0] * 6})
    client.send_message = Mock()
    clock = [0.0]

    def feedback():
        clock[0] += 1.0
        q = np.ones(11) if clock[0] < 13 else np.zeros(11)
        return q, q.copy(), None

    client.get_joint_state = Mock(side_effect=feedback)
    monkeypatch.setattr("emet.controller.zmq_client.timeit.default_timer", lambda: clock[0])
    monkeypatch.setattr("emet.controller.zmq_client.time.sleep", lambda _: None)
    assert client.arm_to([0] * 6, timeout=10) is expected
    assert clock[0] == (14 if expected else 11)


@pytest.mark.parametrize("missing", [False, True])
def test_arm_deadline_remains_bounded_without_progress_or_feedback(missing, monkeypatch):
    client = make_zmq_test_client(StretchZmqClient)
    client._state = {"sim_to_real_ratio": 0.001}  # Existing scale cap is 10x.
    client.in_manipulation_mode = Mock(return_value=True)
    for joint in ("arm", "lift", "base_x", "wrist_roll", "wrist_pitch", "wrist_yaw"):
        setattr(client, f"_{joint}_joint_tolerance", 0.01)
    client.send_action = Mock(return_value={"joint": [0] * 6})
    client.send_message = Mock()
    client.get_joint_state = Mock(return_value=(None, None, None) if missing else (np.ones(11), np.ones(11), None))
    clock = count(0, 1)
    monkeypatch.setattr("emet.controller.zmq_client.timeit.default_timer", lambda: next(clock))
    monkeypatch.setattr("emet.controller.zmq_client.time.sleep", lambda _: None)
    assert client.arm_to([0] * 6, timeout=1) is False
    assert client.get_joint_state.call_count == 11
