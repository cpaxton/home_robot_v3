# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from unittest.mock import Mock

import numpy as np

from emet.controller.zmq_client import StretchZmqClient


def test_arm_retry_reuses_command_identity(monkeypatch):
    client = object.__new__(StretchZmqClient)
    client._finish = False
    client.in_manipulation_mode = Mock(return_value=True)
    for joint in ("arm", "lift", "base_x", "wrist_roll", "wrist_pitch", "wrist_yaw"):
        setattr(client, f"_{joint}_joint_tolerance", 0.01)
    sent = {"joint": [0] * 6, "command": {"sequence": 9}, "step": 12}
    client.send_action = Mock(return_value=sent)
    client.send_message = Mock()
    states = [(np.ones(11), np.ones(11), None)] * 42 + [(np.zeros(11), np.zeros(11), None)]
    client.get_joint_state = Mock(side_effect=states)
    monkeypatch.setattr("emet.controller.zmq_client.time.sleep", lambda _: None)
    assert client.arm_to([0] * 6, blocking=True) is True
    client.send_action.assert_called_once()
    client.send_message.assert_called_once_with(sent)
