# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Interference rejects evidence, never supplies self/free-space labels."""

from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from prepare_observed_stretch_fixture import prepare

from emet.controller.manipulation.encoder_capture import MotorCapture
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_effects import CaptureInterval
from emet.controller.manipulation.robot_view_guard import RobotViewGuard, rigid_gripper_outer_solids
from emet.motion.observed_solids import WorldSolid
from emet.robots.stretch import STRETCH_ARM_CHAIN
from emet.simulation.depth_registration import register_depth
from emet.simulation.synchronized_capture import robot_kinematic_signature


def guard(**overrides):
    transform, pose = np.eye(4), np.eye(4)
    transform[0, 3], pose[:3, 3] = 0.2, [1.0, 2.0, 3.0]
    data = {
        "capture": CaptureInterval("sample", 10.0, 10.001, 10.01, "clock"),
        "raw_depth": np.array([[1.0, 0.0, 2.0]]),
        "depth_k": np.eye(3),
        "rgb_k": np.eye(3),
        "rgb_from_depth": transform,
        "world_from_rgb": pose,
        "output_shape": (1, 3),
        "geom_ids": (0,),
        "outer_solids": (WorldSolid("sphere", (0.1,), np.eye(4)),),
        "model_signature": "robot",
        "calibration_id": "calibration",
        "point_error_m": 0.001,
    }
    data.update(overrides)
    return RobotViewGuard.from_registered_depth(**data)


def test_source_points_use_original_depth_ray_not_registered_rgb_pixel_center():
    layer = guard()
    np.testing.assert_allclose(layer.source_points_world[0, 0], [1.2, 2.0, 4.0])
    assert np.isnan(layer.source_points_world[0, 1]).all()
    assert layer.registered_depth[0, 0] == 1.0
    transform = np.eye(4)
    transform[0, 3] = 0.2
    expected = register_depth([[1.0, 0.0, 2.0]], np.eye(3), np.eye(3), transform, (1, 3))
    np.testing.assert_array_equal(layer.registered_depth, expected)
    with pytest.raises(ValueError):
        layer.source_points_world.setflags(write=True)
    layer.require(layer.capture, layer.registered_depth, layer.registered_points_world, now=10.02)


def test_calibration_fingerprint_ignores_fk_roundoff_but_not_a_real_baseline_change():
    original = guard()
    transform = np.eye(4)
    transform[0, 3] = 0.2 + 1e-15
    transform[1, 3] = -1e-16
    assert guard(rgb_from_depth=transform).profile_identity == original.profile_identity
    transform[0, 3] = 0.21
    assert guard(rgb_from_depth=transform).profile_identity != original.profile_identity


def test_guard_requires_the_same_pinhole_intrinsics_contract_as_capture_arrays():
    k = np.eye(3)
    k[0, 1] = 0.1
    with pytest.raises(ValueError, match="invalid_observed_camera_geometry"):
        guard(rgb_k=k)


def test_wrong_capture_depth_or_uncertainty_cannot_be_reused():
    original = guard()
    with pytest.raises(ObservationUnavailable, match="robot_view_capture_mismatch"):
        original.require(
            replace(original.capture, observation_id="retimed"),
            original.registered_depth,
            original.registered_points_world,
            now=10.02,
        )
    with pytest.raises(ValueError):
        replace(original, outer_solids=(replace(original.outer_solids[0], offset_m=-0.001),))
    with pytest.raises(ValueError):
        replace(original, point_error_m=-0.001)


def test_rigid_geometry_scope_uses_motors_and_never_invents_passive_measurements(tmp_path):
    prepare(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "stretch.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, STRETCH_ARM_CHAIN.ee_body)
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = data.xmat[ee].reshape(3, 3), data.xpos[ee]
    ticket = CaptureInterval("sample", 10.0, 10.001, 10.01, "clock")
    estimate = SimpleNamespace(
        world_from_ee=pose,
        motor_capture=MotorCapture(ticket, np.zeros(8), robot_kinematic_signature(model)),
        nominal_qpos=np.full(model.nq, np.nan),
    )
    ids, solids, signature = rigid_gripper_outer_solids(
        model, estimate, now=10.02, position_error_m=0.001, rotation_error_rad=0.005
    )
    bodies = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) for g in ids}
    assert "link_d405" in bodies and "link_SG3_gripper_body" in bodies
    assert not any("finger" in name or "rubber" in name for name in bodies)
    assert solids and signature
    estimate.nominal_qpos[:] = 99.0
    again = rigid_gripper_outer_solids(model, estimate, now=10.02, position_error_m=0.001, rotation_error_rad=0.005)
    for a, b in zip(solids, again[1], strict=True):
        np.testing.assert_array_equal(a.world_from_local, b.world_from_local)
