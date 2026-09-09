# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import numpy as np
import pytest

from emet.motion.arm_manip_profile import ArmManipProfile
from emet.robots.nori import (
    NORI_CAMERA_NAMES,
    NORI_JOINT_NAMES,
    _nori_arm_chain,
)


def test_nori_spec_and_registry():
    from emet.robots import get_robot_backend, get_robot_spec

    spec = get_robot_spec("nori")
    assert spec is not None and spec.name == "nori"
    assert spec.mjcf_path and spec.mjcf_path.endswith("nori_a3/nori_a3.xml")
    assert spec.dof == len(NORI_JOINT_NAMES)
    assert set(spec.camera_names) == set(NORI_CAMERA_NAMES)
    assert spec.arm_chains["left"].ee_body == "left_wrist_roll_link"
    assert spec.advertise_kinematic_manip is True
    # alias resolves to the same backend
    assert get_robot_spec("nori_a3").name == "nori"
    from emet.robots.nori import NoriBackend

    assert isinstance(get_robot_backend("nori"), NoriBackend)


def test_nori_arm_chains_per_side():
    left = _nori_arm_chain("left")
    right = _nori_arm_chain("right")
    assert len(left.joint_names) == len(right.joint_names) == 8  # lift + 7 arm joints
    assert left.joint_names[0] == "lift_extension_joint"  # lift lowers to floor objects
    assert left.ee_body == "left_wrist_roll_link"
    assert right.ee_body == "right_wrist_roll_link"
    assert all(j.startswith("left_") for j in left.joint_names[1:])
    assert all(j.startswith("right_") for j in right.joint_names[1:])
    assert "left_gripper_idler_joint" not in left.joint_names  # mimics excluded
    assert len(left.link_bodies) > 1 and left.link_bodies[-1] == left.ee_body
    assert len(right.link_bodies) > 1 and right.link_bodies[-1] == right.ee_body


def test_nori_profiles_resolve():
    for side in ("left", "right"):
        p = ArmManipProfile.for_robot("nori", arm=side)
        assert tuple(p.joint_names) == _nori_arm_chain(side).joint_names
        assert p.ee_body == f"{side}_wrist_roll_link"
        assert tuple(p.actuator_names) == _nori_arm_chain(side).actuator_names
        assert len(p.home_cmd) == len(p.actuator_names)
        assert p.gripper_contact_bodies()
        assert len(p.link_bodies) > 1


def test_nori_mjcf_mimics_and_named_actuators():
    pytest.importorskip("mujoco")
    import mujoco

    from emet.robots import get_robot_spec

    spec = get_robot_spec("nori")
    model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
    assert int(model.neq) >= 3
    for jn in (
        "left_gripper_idler_joint",
        "right_gripper_idler_joint",
        "lift_middle_joint",
    ):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn) >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jn) < 0
    for an in spec.actuator_names:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, an) >= 0, an


def test_nori_ik_reaches_target():
    pytest.importorskip("mujoco")
    import mujoco

    from emet.robots import get_robot_spec

    spec = get_robot_spec("nori")
    model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    p = ArmManipProfile.for_robot("nori", arm="left")
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p.ee_body)
    assert ee_id >= 0
    target = np.array(data.body(ee_id).xpos, dtype=np.float64).copy()
    target[2] -= 0.05  # reach down a few cm
    target[0] += 0.03

    from emet.motion.mujoco_arm_ik import solve_position_ik_multiseed

    res = solve_position_ik_multiseed(
        model,
        data,
        ee_body=p.ee_body,
        joint_names=list(p.joint_names),
        target_pos=target,
        tol_m=0.05,
        max_iters=200,
    )
    assert res.success, f"nori left IK failed (err {res.pos_error_m:.4f} m)"
    assert res.pos_error_m <= 0.06


def test_nori_capability_gate():
    from emet.simulation.robosuite_server import _robot_supports_kinematic_manip

    assert _robot_supports_kinematic_manip("nori") is True
    assert _robot_supports_kinematic_manip("nori_a3") is True


def test_nori_create_model():
    from emet.robots import get_robot_backend

    backend = get_robot_backend("nori")
    model = backend.create_model()
    assert model.get_dof() == 21
    assert model.get_footprint() is not None