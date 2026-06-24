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

import mujoco
import numpy as np

from emet.robots.xlerobot import (
    XLEROBOT_HEAD_ACTUATORS,
    XLEROBOT_JAW_OPEN,
    XLeRobotBackend,
    jaw_angle_from_normalized,
    jaw_normalized_from_angle,
)
from emet.simulation.gripper_action import apply_gripper_action_robosuite
from emet.simulation.head_look_action import apply_head_to_robosuite


def test_xlerobot_mjcf_has_head_actuators():
    spec = XLeRobotBackend().get_spec()
    model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
    assert model.nu == 20
    assert len(spec.actuator_names) == 20
    for aname in XLEROBOT_HEAD_ACTUATORS:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname) >= 0


def test_xlerobot_head_to_sets_ctrl():
    spec = XLeRobotBackend().get_spec()
    model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
    data = mujoco.MjData(model)
    n = apply_head_to_robosuite(spec, model, data, pan=0.4, tilt=-0.2)
    assert n == 2
    pan_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_pan")
    tilt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "head_tilt")
    assert float(data.ctrl[pan_id]) == 0.4
    assert float(data.ctrl[tilt_id]) == -0.2


def test_xlerobot_dual_gripper_actions():
    spec = XLeRobotBackend().get_spec()
    model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
    data = mujoco.MjData(model)

    left = apply_gripper_action_robosuite(spec, model, data, {"gripper_left": 1.0})
    assert "Jaw_L" in left
    right = apply_gripper_action_robosuite(spec, model, data, {"gripper_right": 0.0})
    assert "Jaw_R" in right

    lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Jaw_L")
    rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Jaw_R")
    assert float(data.ctrl[lid]) == jaw_angle_from_normalized(1.0)
    assert float(data.ctrl[rid]) == jaw_angle_from_normalized(0.0)


def test_xlerobot_home_keyframe_name():
    from emet.simulation.robosuite_load_utils import resolve_robot_home_keyframe_id, robot_home_keyframe_name

    spec = XLeRobotBackend().get_spec()
    assert robot_home_keyframe_name(spec) == "xlerobot_home"
    model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
    assert resolve_robot_home_keyframe_id(model, spec) >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home") < 0


def test_xlerobot_jaw_normalized_roundtrip():
    angle = jaw_angle_from_normalized(0.75)
    assert np.isclose(jaw_normalized_from_angle(angle), 0.75)
    assert jaw_angle_from_normalized(0.0) == 0.0
    assert jaw_angle_from_normalized(1.0) == XLEROBOT_JAW_OPEN


def test_xlerobot_apply_home_keyframe_preserving_planar_base():
    from emet.simulation.robosuite_load_utils import apply_home_keyframe_preserving_planar_base

    spec = XLeRobotBackend().get_spec()
    model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
    data = mujoco.MjData(model)
    planar = spec.planar_base_joint_names
    assert planar is not None
    saved = []
    for jname in planar:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        qadr = int(model.jnt_qposadr[jid])
        data.qpos[qadr] = 0.2 + 0.05 * len(saved)
        saved.append(float(data.qpos[qadr]))
    mujoco.mj_forward(model, data)
    assert apply_home_keyframe_preserving_planar_base(
        model,
        data,
        planar_joint_names=planar,
        base_body_name=spec.base_link_name,
        spec=spec,
    )
    for jname, val in zip(planar, saved, strict=True):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        qadr = int(model.jnt_qposadr[jid])
        assert abs(float(data.qpos[qadr]) - val) < 1e-6


def test_xlerobot_gripper_legacy_stretch_key_targets_left_jaw():
    spec = XLeRobotBackend().get_spec()
    model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
    data = mujoco.MjData(model)
    updated = apply_gripper_action_robosuite(spec, model, data, {"gripper": 1.0})
    assert updated == ["Jaw_L"]
    lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Jaw_L")
    assert float(data.ctrl[lid]) == jaw_angle_from_normalized(1.0)
