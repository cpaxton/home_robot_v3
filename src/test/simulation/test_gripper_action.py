# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Unit tests for dual-gripper ZMQ → MuJoCo mapping (XLeRobot)."""

from __future__ import annotations

import mujoco

from emet.robots.franka_fr3 import FrankaFR3Backend
from emet.robots.xlerobot import (
    XLeRobotBackend,
    jaw_angle_from_normalized,
    parse_xlerobot_gripper_side,
)
from emet.simulation.gripper_action import apply_gripper_action_robosuite


def test_gripper_action_ignored_for_non_xlerobot():
    spec = FrankaFR3Backend().get_spec()
    model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
    data = mujoco.MjData(model)
    assert apply_gripper_action_robosuite(spec, model, data, {"gripper_left": 1.0}) == []


def test_gripper_left_right_and_legacy_gripper_key():
    spec = XLeRobotBackend().get_spec()
    model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
    data = mujoco.MjData(model)

    both = apply_gripper_action_robosuite(
        spec,
        model,
        data,
        {"gripper_left": 1.0, "gripper_right": 0.5, "left_gripper": 0.25},
    )
    assert "Jaw_L" in both
    assert "Jaw_R" in both

    data2 = mujoco.MjData(model)
    legacy = apply_gripper_action_robosuite(spec, model, data2, {"gripper": 0.8})
    assert legacy == ["Jaw_L"]
    lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Jaw_L")
    assert float(data2.ctrl[lid]) == jaw_angle_from_normalized(0.8)


def test_parse_xlerobot_gripper_side_aliases():
    assert parse_xlerobot_gripper_side("left_gripper") == "left"
    assert parse_xlerobot_gripper_side("right_gripper") == "right"
    assert parse_xlerobot_gripper_side("gripper_right") == "right"
    assert parse_xlerobot_gripper_side("gripper") == "left"
