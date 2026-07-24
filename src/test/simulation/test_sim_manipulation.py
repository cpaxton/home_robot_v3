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

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from emet.simulation.sim_manipulation import freejoint_ancestor_body_id, set_free_body_pose


def test_set_free_body_pose_moves_object2():
    scene = Path(__file__).resolve().parents[2] / "emet" / "assets" / "robot" / "scene_environment.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    before = np.array(data.body("object2").xpos, dtype=np.float64).copy()
    target = before + np.array([0.0, 0.0, 0.12])
    assert set_free_body_pose(model, data, "object2", target)
    after = np.array(data.body("object2").xpos, dtype=np.float64)
    assert np.linalg.norm(after - target) < 1e-5


def test_set_free_body_pose_via_molmo_child_if_available():
    """Molmo iTHOR objects: teleport mesh child by moving freejoint parent."""
    scene = Path.home() / ".cache/molmospaces/assets/scenes/ithor/FloorPlan1_physics.xml"
    if not scene.is_file():
        pytest.skip("MolmoSpaces iTHOR FloorPlan1 not installed")
    child = "apple_038a0ea9b393da66a161da588e6ecc2a_1_1_0"
    parent = "apple_038a0ea9b393da66a161da588e6ecc2a_1_0_0"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    child_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, child)
    parent_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, parent)
    assert child_id >= 0 and parent_id >= 0
    assert freejoint_ancestor_body_id(model, child_id) == parent_id
    before = np.array(data.body(child_id).xpos, dtype=np.float64).copy()
    target = before + np.array([0.05, 0.0, 0.12])
    assert set_free_body_pose(model, data, child, target)
    after = np.array(data.body(child_id).xpos, dtype=np.float64)
    assert np.linalg.norm(after - target) < 1e-4


def test_robot_sim_body_pose_teleport_supported_caps():
    from emet.simulation.sim_manipulation import (
        can_use_sim_gt_manip,
        prefer_kinematic_manip,
        prefer_sim_teleport_manip,
        robot_sim_body_pose_teleport_supported,
    )

    class _R:
        def __init__(self, session):
            self._s = session

        def get_emet_session(self):
            return self._s

    assert not robot_sim_body_pose_teleport_supported(_R(None))
    assert not robot_sim_body_pose_teleport_supported(_R({"is_simulation": True, "capabilities": {}}))
    assert robot_sim_body_pose_teleport_supported(
        _R({"is_simulation": True, "capabilities": {"sim_set_body_pose": True}})
    )
    robot = _R({"is_simulation": True, "capabilities": {"sim_set_body_pose": True}})
    assert prefer_sim_teleport_manip(robot, visual_servo=False)
    assert not prefer_sim_teleport_manip(robot, visual_servo=True)

    # kinematic without server cap → not prefer_kinematic, but can still use teleport GT
    assert not prefer_kinematic_manip(robot, manip_mode="kinematic", visual_servo=False)
    assert can_use_sim_gt_manip(robot, manip_mode="kinematic", visual_servo=False)

    kin = _R(
        {
            "is_simulation": True,
            "capabilities": {"sim_set_body_pose": True, "kinematic_manip": True},
        }
    )
    assert prefer_kinematic_manip(kin, manip_mode="kinematic", visual_servo=False)
    assert can_use_sim_gt_manip(kin, manip_mode="kinematic", visual_servo=False)
    assert not prefer_kinematic_manip(kin, manip_mode="kinematic", visual_servo=True)
    assert not can_use_sim_gt_manip(kin, manip_mode="kinematic", visual_servo=True)
