# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import numpy as np
import pytest

from emet.controller.generic_zmq_client import GenericZmqClient
from emet.robots import get_robot_spec


@pytest.mark.parametrize("robot", ["rby1", "galaxea_r1", "xlerobot"])
def test_look_feedback_uses_measured_adapter_joint_mapping(robot):
    client = GenericZmqClient.__new__(GenericZmqClient)
    client._spec = get_robot_spec(robot)
    client._joint_index = {name: i for i, name in enumerate(client._spec.joint_names)}
    positions = np.zeros(len(client._joint_index))
    for name, value in zip(client._spec.look_joint_names, [0.3, -0.8], strict=True):
        positions[client._joint_index[name]] = value
    client.get_joint_state = lambda **_: (positions, None, None)
    np.testing.assert_allclose(client.get_pan_tilt(), [0.3, -0.8])
    client.get_joint_state = lambda **_: (None, None, None)
    assert np.isnan(client.get_pan_tilt()).all()
