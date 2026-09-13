# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from pathlib import Path

import mujoco
import numpy as np
import pytest


@pytest.mark.parametrize(
    "scene,neighbor_x", [("scene_clearance_control.xml", -0.25), ("scene_right_neighbor.xml", 0.18)]
)
def test_control_changes_only_neighbor_pose(scene, neighbor_x):
    assets = Path(__file__).resolve().parents[2] / "emet/assets/robot"
    original = mujoco.MjModel.from_xml_path(str(assets / "scene.xml"))
    control = mujoco.MjModel.from_xml_path(str(assets / scene))
    assert original.names == control.names
    for field in ("geom_size", "body_mass", "body_inertia", "actuator_gainprm", "actuator_ctrlrange", "jnt_range"):
        np.testing.assert_array_equal(getattr(original, field), getattr(control, field))
    expected = original.qpos0.copy()
    addr = original.jnt_qposadr[original.body("object1").jntadr[0]]
    expected[addr] = neighbor_x
    np.testing.assert_array_equal(control.qpos0, expected)
    expected_pos = original.body_pos.copy()
    expected_pos[original.body("object1").id, 0] = neighbor_x
    np.testing.assert_array_equal(control.body_pos, expected_pos)
