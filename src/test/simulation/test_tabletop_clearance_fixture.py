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


def test_noslip_ablation_only_changes_solver_iterations():
    assets = Path(__file__).resolve().parents[2] / "emet/assets/robot"
    control = mujoco.MjModel.from_xml_path(str(assets / "scene_right_neighbor.xml"))
    candidate = mujoco.MjModel.from_xml_path(str(assets / "scene_right_neighbor_noslip.xml"))
    assert control.names == candidate.names
    for field in (
        "qpos0",
        "body_pos",
        "body_quat",
        "body_mass",
        "body_inertia",
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "geom_contype",
        "geom_conaffinity",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "jnt_range",
        "dof_damping",
        "dof_frictionloss",
        "eq_data",
        "eq_solref",
        "eq_solimp",
    ):
        np.testing.assert_array_equal(getattr(candidate, field), getattr(control, field))
    assert control.opt.noslip_iterations == 0 and candidate.opt.noslip_iterations == 10
    for field in (
        "timestep",
        "solver",
        "cone",
        "impratio",
        "integrator",
        "iterations",
        "tolerance",
        "noslip_tolerance",
    ):
        assert getattr(candidate.opt, field) == getattr(control.opt, field)
