# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Analytic gripper envelopes versus independently evaluated MuJoCo geometry."""

from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from prepare_observed_stretch_fixture import prepare

from emet.controller.manipulation.encoder_capture import MotorCapture
from emet.controller.manipulation.gripper_geometry import bound_stretch_gripper
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_effects import CaptureInterval
from emet.motion.placement_geometry import box_corners
from emet.robots.stretch import STRETCH_ARM_CHAIN
from emet.simulation.synchronized_capture import robot_kinematic_signature


@pytest.fixture
def setup(tmp_path):
    prepare(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "stretch.xml"))
    data = mujoco.MjData(model)
    data.joint("joint_lift").qpos[0] = 0.7
    data.joint("joint_wrist_pitch").qpos[0] = -0.3
    data.joint("joint_gripper_slide").qpos[0] = 0.02
    for i in range(4):
        data.joint(f"joint_arm_l{i}").qpos[0] = 0.05
    mujoco.mj_kinematics(model, data)
    ee = data.body(STRETCH_ARM_CHAIN.ee_body)
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = ee.xmat.reshape(3, 3), ee.xpos
    values = np.array([0.7, 0.2, 0.0, -0.3, 0.0, 0.02, 0.0, 0.0])
    capture = CaptureInterval("capture", 10.0, 10.001, 10.01, "calibration")
    motors = MotorCapture(capture, values, robot_kinematic_signature(model))
    estimate = SimpleNamespace(world_from_ee=pose, motor_capture=motors, nominal_qpos=data.qpos.copy())
    return model, data, estimate


def geom_corners(model, data, gid):
    aabb = model.geom_aabb[gid]
    return (
        box_corners([aabb[:3] - aabb[3:], aabb[:3] + aabb[3:]]) @ data.geom_xmat[gid].reshape(3, 3).T
        + data.geom_xpos[gid]
    )


def test_full_joint_envelope_contains_random_passive_states_and_telescope_splits(setup):
    model, data, estimate = setup
    bound = bound_stretch_gripper(model, estimate)
    assert any(model.geom_bodyid[g] == model.body("link_SG3_gripper_body").id for g in bound.geom_ids)
    rng = np.random.default_rng(17)
    for _ in range(200):
        for name, lo, hi in bound.joint_intervals:
            data.joint(name).qpos[0] = rng.uniform(lo, hi)
        split = rng.dirichlet(np.ones(4)) * 0.2
        for i in range(4):
            data.joint(f"joint_arm_l{i}").qpos[0] = split[i]
        mujoco.mj_kinematics(model, data)
        for gid, box in zip(bound.geom_ids, bound.bounds_world, strict=True):
            corners = geom_corners(model, data, gid)
            assert np.all(corners >= box[0] - 1e-9) and np.all(corners <= box[1] + 1e-9)
    assert not bound.execution_qualified
    before, after = bound.robot_brackets(estimate, model, now=10.1)
    assert before.joints.shape == (8,) and after.joints.shape == (8,)
    assert before.gripper_bounds is bound.bounds_world or np.array_equal(before.gripper_bounds, bound.bounds_world)
    assert before.source == "measured_joint_feedback"
    with pytest.raises(ValueError):
        bound.bounds_world.setflags(write=True)


def test_geometry_can_use_explicit_uncertainty_but_never_reads_passive_qpos(setup):
    model, data, estimate = setup
    limits = dict.fromkeys(("joint_gripper_finger_left_open", "joint_gripper_finger_right_open"), (0.1, 0.3))
    with pytest.raises(ValueError, match="calibration_required"):
        bound_stretch_gripper(model, estimate, joint_intervals=limits)
    full = bound_stretch_gripper(model, estimate)
    narrowed = bound_stretch_gripper(model, estimate, joint_intervals=limits, calibration_id="declared-test-only")
    assert "declared_intervals:declared-test-only" in narrowed.assumptions
    assert np.sum(narrowed.bounds_world[:, 1] - narrowed.bounds_world[:, 0]) < np.sum(
        full.bounds_world[:, 1] - full.bounds_world[:, 0]
    )
    estimate.nominal_qpos[:] = 999.0
    same = bound_stretch_gripper(model, estimate)
    np.testing.assert_array_equal(same.bounds_world, full.bounds_world)
    # A pose-error budget grows the actual output volumes; it is not metadata only.
    inflated = bound_stretch_gripper(model, estimate, position_error_m=0.003, rotation_error_rad=0.01)
    assert np.all(inflated.bounds_world[:, 0] < full.bounds_world[:, 0])
    assert np.all(inflated.bounds_world[:, 1] > full.bounds_world[:, 1])


def test_stale_mismatched_capture_or_changed_geometry_cannot_be_consumed(setup):
    model, data, estimate = setup
    bound = bound_stretch_gripper(model, estimate)
    with pytest.raises(ObservationUnavailable):
        bound.robot_brackets(estimate, model, now=13.0)
    other = replace(estimate.motor_capture, capture=replace(estimate.motor_capture.capture, observation_id="other"))
    wrong = SimpleNamespace(world_from_ee=estimate.world_from_ee, motor_capture=other)
    with pytest.raises(ObservationUnavailable, match="changed"):
        bound.robot_brackets(wrong, model, now=10.1)
    positions = estimate.motor_capture.positions.copy()
    positions[5] += 0.01  # EE is unchanged by aperture; bind the actual motors too.
    altered = SimpleNamespace(
        world_from_ee=estimate.world_from_ee, motor_capture=replace(estimate.motor_capture, positions=positions)
    )
    with pytest.raises(ObservationUnavailable, match="changed"):
        bound.robot_brackets(altered, model, now=10.1)
    model.geom_aabb[bound.geom_ids[0], 3] += 0.01
    with pytest.raises(ObservationUnavailable, match="changed"):
        bound.robot_brackets(estimate, model, now=10.1)


def test_measured_aperture_and_unknown_interval_names_cannot_be_overridden(setup):
    model, data, estimate = setup
    with pytest.raises(ValueError, match="override"):
        bound_stretch_gripper(
            model, estimate, joint_intervals={"joint_gripper_slide": (0.0, 0.04)}, calibration_id="test"
        )
    with pytest.raises(ValueError, match="unknown_gripper"):
        bound_stretch_gripper(model, estimate, joint_intervals={"not-a-joint": (0.0, 1.0)}, calibration_id="test")
