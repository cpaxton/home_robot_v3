# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Whole-robot enclosures against independent FK at compatible hidden states."""

import xml.etree.ElementTree as ET
from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from prepare_observed_stretch_fixture import prepare

from emet.controller.manipulation.encoder_capture import MotorCapture
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_effects import CaptureInterval
from emet.controller.manipulation.stretch_geometry import bound_stretch_robot
from emet.motion.placement_geometry import box_corners
from emet.simulation.synchronized_capture import robot_kinematic_signature


@pytest.fixture
def setup(tmp_path):
    prepare(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "stretch.xml"))
    capture = CaptureInterval("motors", 10.0, 10.001, 10.01, "clock")
    values = np.array([0.7, 0.2, 0.4, -0.3, 0.2, 0.02, -0.7, -0.5])
    motors = MotorCapture(capture, values, robot_kinematic_signature(model))
    pose = np.eye(4)
    angle = 0.3
    pose[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    pose[:3, 3] = [0.2, -0.3, 0.01]
    return model, SimpleNamespace(motor_capture=motors, world_from_base=pose)


def bound(model, estimate, **kwargs):
    return bound_stretch_robot(model, estimate, now=10.1, position_error_m=0.0, rotation_error_rad=0.0, **kwargs)


def test_all_collidable_geoms_enclosed_for_passive_states_and_unequal_telescope_splits(setup):
    model, estimate = setup
    envelope = bound(model, estimate)
    assert set(envelope.geom_ids) == {
        g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]
    }
    data = mujoco.MjData(model)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, estimate.world_from_base[:3, :3].ravel())
    data.qpos[:7] = np.r_[estimate.world_from_base[:3, 3], quat]
    rng = np.random.default_rng(43)
    for _ in range(100):
        for name, low, high in envelope.joint_intervals:
            data.joint(name).qpos[0] = rng.uniform(low, high)
        split = rng.dirichlet(np.ones(4)) * 0.2
        while np.any(split > 0.13):
            split = rng.dirichlet(np.ones(4)) * 0.2
        for i in range(4):
            data.joint(f"joint_arm_l{i}").qpos[0] = split[i]
        mujoco.mj_kinematics(model, data)
        for g, box in zip(envelope.geom_ids, envelope.bounds_world, strict=True):
            aabb = model.geom_aabb[g]
            corners = box_corners([aabb[:3] - aabb[3:], aabb[:3] + aabb[3:]])
            world = corners @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            assert np.all(world >= box[0] - 1e-8), (g, "low")
            assert np.all(world <= box[1] + 1e-8), (g, "high")
    assert not envelope.execution_qualified
    assert "measured_total_extension" in envelope.assumptions
    with pytest.raises(ValueError):
        envelope.bounds_world.setflags(write=True)


def test_telescope_intervals_use_total_and_bound_distal_geometry_independently_of_split(setup):
    model, estimate = setup
    small = estimate.motor_capture.positions.copy()
    small[1] = 0.015
    estimate.motor_capture = replace(estimate.motor_capture, positions=small)
    envelope = bound(model, estimate)
    for name, low, high in envelope.joint_intervals:
        if name.startswith("joint_arm_l"):
            assert low == 0 and high == pytest.approx(0.015)
    # Nominal passive/debug state is neither accepted nor consumed.
    estimate.nominal_qpos = np.full(model.nq, np.nan)
    np.testing.assert_array_equal(envelope.bounds_world, bound(model, estimate).bounds_world)
    inflated = bound_stretch_robot(model, estimate, now=10.1, position_error_m=0.002, rotation_error_rad=0.01)
    assert np.all(inflated.bounds_world[:, 0] < envelope.bounds_world[:, 0])
    assert np.all(inflated.bounds_world[:, 1] > envelope.bounds_world[:, 1])


def test_unavailable_stale_or_changed_motor_geometry_rejects(setup):
    model, estimate = setup
    with pytest.raises(ObservationUnavailable, match="stale"):
        bound_stretch_robot(model, estimate, now=13.0, position_error_m=0.0, rotation_error_rad=0.0)
    invalid = estimate.motor_capture.positions.copy()
    invalid[1] = 0.6
    estimate.motor_capture = replace(estimate.motor_capture, positions=invalid)
    with pytest.raises(ObservationUnavailable, match="interval_unavailable"):
        bound(model, estimate)
    model.jnt_axis[model.joint("joint_arm_l0").id] = [1.0, 0.0, 0.0]
    with pytest.raises(ObservationUnavailable, match="model_mismatch"):
        bound(model, estimate)


def test_changed_telescope_transmission_invalidates_the_outer_bound(setup):
    model, estimate = setup
    tendon = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "extend")
    model.wrap_prm[int(model.tendon_adr[tendon])] = 2.0
    with pytest.raises(ObservationUnavailable, match="robot_capture_model_mismatch"):
        bound(model, estimate)


@pytest.mark.parametrize("total", [0.0, 0.52])
def test_arm_and_motor_limit_endpoints_remain_finite(setup, total):
    model, estimate = setup
    values = estimate.motor_capture.positions.copy()
    values[1] = total
    for motor, index in (
        ("lift", 0),
        ("wrist_yaw", 2),
        ("wrist_pitch", 3),
        ("wrist_roll", 4),
        ("gripper", 5),
        ("head_pan", 6),
        ("head_tilt", 7),
    ):
        name = "joint_gripper_slide" if motor == "gripper" else "joint_" + motor
        values[index] = model.jnt_range[model.joint(name).id, int(total > 0)]
    estimate.motor_capture = replace(estimate.motor_capture, positions=values)
    envelope = bound(model, estimate)
    assert np.isfinite(envelope.bounds_world).all()
    assert np.all(envelope.bounds_world[:, 0] < envelope.bounds_world[:, 1])
    for name, low, high in envelope.joint_intervals:
        if name.startswith("joint_arm_l"):
            assert low == pytest.approx(total / 4)
            assert high == pytest.approx(total / 4)


def test_nonrigid_base_localization_cannot_create_a_bound(setup):
    model, estimate = setup
    estimate.world_from_base[0, 0] = 3.0
    with pytest.raises(ValueError):
        bound(model, estimate)


def test_changed_arm_reference_position_invalidates_motor_feedback(setup):
    model, estimate = setup
    model.qpos0[model.jnt_qposadr[model.joint("joint_arm_l0").id]] = 0.01
    with pytest.raises(ObservationUnavailable, match="robot_capture_model_mismatch"):
        bound(model, estimate)


def test_nonserial_named_telescope_cannot_use_a_distal_sum_override(setup, tmp_path):
    _, estimate = setup
    xml = ET.parse(tmp_path / "stretch.xml")
    parents = {child: parent for parent in xml.iter() for child in parent}
    joint = xml.find('.//joint[@name="joint_arm_l0"]')
    body = parents[joint]
    parents[body].remove(body)
    xml.find('.//body[@name="base_link"]').append(body)
    path = tmp_path / "nonserial.xml"
    xml.write(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    # A matching new profile cannot turn parallel but nonserial slides into a
    # serial total-extension measurement.
    estimate.motor_capture = replace(estimate.motor_capture, kinematic_signature=robot_kinematic_signature(model))
    with pytest.raises(ObservationUnavailable, match="robot_capture_model_mismatch"):
        bound(model, estimate)
