# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from pathlib import Path
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest


@pytest.mark.parametrize("payload_kg", [0.0, 0.5, 0.75])
def test_lift_holds_payload_without_relaxing_position_or_force_limits(payload_kg):
    model = mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[2] / "emet/assets/robot/scene.xml"))
    # Evaluator-only static payload control, not a learned grasp or a policy input.
    payload = model.body("link_grasp_center")
    payload.mass = payload_kg
    payload.inertia = np.full(3, payload_kg * 0.001)
    model.body_gravcomp[payload.id] = 0
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    lift_index = model.joint("joint_lift").qposadr[0]
    data.qpos[lift_index] = 0.5
    data.ctrl[model.actuator("lift").id] = 0.5
    data.ctrl[model.actuator("arm").id] = 0.15
    data.ctrl[model.actuator("wrist_pitch").id] = -0.4
    for _ in range(round(2 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    target = 0.8
    data.ctrl[model.actuator("lift").id] = target
    peak = 0
    for _ in range(round(2.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        peak = max(peak, data.qpos[lift_index])
        assert abs(data.actuator("lift").force[0]) <= 70.00001
    assert abs(data.qpos[lift_index] - target) < 0.02
    assert peak <= target + 0.02


@pytest.mark.parametrize("posture", ["navigation", "manipulation"])
def test_posture_preserves_gripper_command(posture):
    from emet.simulation.mujoco_server_stretch import MujocoZmqServer

    server = object.__new__(MujocoZmqServer)
    server.manip_to = Mock()
    assert server.set_posture(posture)
    assert server.manip_to.call_args.kwargs == {"all_joints": True, "skip_gripper": True}
