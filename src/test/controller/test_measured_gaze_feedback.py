# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.controller.generic_zmq_client import GenericZmqClient
from emet.robots import ROBOT_REGISTRY, get_robot_spec
from emet.robots.galaxea_r1 import GalaxeaR1Backend
from emet.robots.rby1 import Rby1Backend


@pytest.mark.parametrize("backend", [Rby1Backend, GalaxeaR1Backend])
def test_proxy_gaze_reads_actual_declared_joint_state(backend):
    client = GenericZmqClient.__new__(GenericZmqClient)
    client._spec = backend().get_spec()
    client._joint_index = {name: i for i, name in enumerate(client._spec.joint_names)}
    q = np.zeros(len(client._spec.joint_names))
    q[client._joint_index["torso_joint4"]] = 0.3
    q[client._joint_index["torso_joint3"]] = -0.6
    client.get_joint_state = Mock(return_value=(q, None, None))
    assert client.get_pan_tilt() == (0.3, -0.6)


@pytest.mark.parametrize("failure", ["undeclared", "missing_joint", "missing_state", "nonfinite"])
def test_unavailable_gaze_never_returns_fabricated_zero(failure):
    client = GenericZmqClient.__new__(GenericZmqClient)
    client._spec = SimpleNamespace(look_joint_names=None if failure == "undeclared" else ("pan", "tilt"))
    client._joint_index = {"pan": 0, "tilt": 1} if failure != "missing_joint" else {}
    client._decoded_observation_snapshot = Mock(return_value={})
    q = None if failure == "missing_state" else np.array([0.0, np.nan if failure == "nonfinite" else 0.0])
    client.get_joint_state = Mock(return_value=(q, None, None))
    with pytest.raises(RuntimeError, match="gaze"):
        client.get_pan_tilt()


def test_explicit_fixed_axis_is_zero_but_other_axis_is_measured():
    client = GenericZmqClient.__new__(GenericZmqClient)
    client._spec = SimpleNamespace(look_joint_names=(None, "tilt"))
    client._joint_index = {"tilt": 0}
    client.get_joint_state = Mock(return_value=(np.array([-0.4]), None, None))
    assert client.get_pan_tilt() == (0.0, -0.4)


@pytest.mark.parametrize("robot_name", sorted(ROBOT_REGISTRY))
def test_registered_adapter_gaze_contract(robot_name):
    if robot_name in {"mobile_aloha", "yor"}:
        with pytest.raises(NotImplementedError, match="stub"):
            get_robot_spec(robot_name)
        return  # Enumerated unsupported registry entries, not skipped qualification.
    spec = get_robot_spec(robot_name)
    assert spec is not None
    if spec.name == "stretch":
        from emet.controller.zmq_client import StretchZmqClient
        from emet.motion.kinematics import HelloStretchIdx

        client = StretchZmqClient.__new__(StretchZmqClient)
        q = np.zeros(11)
        q[HelloStretchIdx.HEAD_PAN], q[HelloStretchIdx.HEAD_TILT] = 0.2, -0.3
        client.get_joint_state = Mock(return_value=(q, None, None))
        assert client.get_pan_tilt() == (0.2, -0.3)
        return
    client = GenericZmqClient.__new__(GenericZmqClient)
    client._spec = spec
    client._joint_index = {name: i for i, name in enumerate(spec.joint_names)}
    named_feedback = {"joint_head": -0.3}
    client._decoded_observation_snapshot = Mock(return_value=named_feedback)
    q = np.arange(len(spec.joint_names), dtype=float) / 100
    client.get_joint_state = Mock(return_value=(q, None, None))
    if spec.look_joint_names is None:
        with pytest.raises(RuntimeError, match="unavailable"):
            client.get_pan_tilt()
    else:
        assert all(
            name is None or name in client._joint_index or name in named_feedback for name in spec.look_joint_names
        )
        expected = tuple(
            0.0
            if name is None
            else q[client._joint_index[name]]
            if name in client._joint_index
            else named_feedback[name]
            for name in spec.look_joint_names
        )
        assert client.get_pan_tilt() == expected
