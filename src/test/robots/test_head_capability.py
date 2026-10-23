# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from unittest.mock import Mock

import mujoco
import numpy as np
import pytest

from emet.robots.head_capability import STRETCH_LEGACY_HEAD, HeadCapability, mujoco_head_capability


def test_effective_limits_intersect_active_joint_and_actuator():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><compiler angle="radian"/><worldbody><body>
    <joint name="pan" range="-3.14 1"/><geom size=".1"/>
    <body><joint name="tilt" axis="0 1 0" range="-1.53 .79"/><geom size=".1"/></body>
    </body></worldbody><actuator><position joint="pan" ctrlrange="-2 .8"/>
    <position joint="tilt" ctrlrange="-1.4 .78"/></actuator></mujoco>""")
    capability = mujoco_head_capability(model, ("pan", "tilt"))
    assert capability.pan == (-2, 0.8)
    assert capability.tilt == (-1.4, 0.78)
    assert capability.contains([0, 0.756])
    assert not capability.contains([0, 0.79])
    assert not capability.contains([0])


@pytest.mark.parametrize("caps", [None, {}, {"head_motion": {"pan": [0, 1], "tilt": [1, -1], "source": "bad"}}])
def test_malformed_capabilities_do_not_expand_limits(caps):
    assert HeadCapability.from_session({"capabilities": caps}) is None


@pytest.mark.parametrize("simulation", [False, True])
def test_stretch_commands_and_preflight_share_effective_limits(simulation):
    from emet.controller.zmq_client import StretchZmqClient

    client = StretchZmqClient.__new__(StretchZmqClient)
    head = HeadCapability((-2, 0.7), (-1.53, 0.79), "active_mujoco_model")
    client.get_emet_session = lambda: {
        "runtime_kind": "stretch_mujoco_sim",
        "is_simulation": simulation,
        "capabilities": {"head_motion": head.as_dict()},
    }
    client.send_action = Mock(return_value={"step": 1})
    capability = client.get_head_capability()
    assert capability == (head if simulation else STRETCH_LEGACY_HEAD)
    client.head_to(0, 0.756, blocking=False)
    np.testing.assert_allclose(client.send_action.call_args.args[0]["head_to"], [0, 0.756 if simulation else 0])
