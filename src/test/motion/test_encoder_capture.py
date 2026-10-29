# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Actual motor readouts remain separate from unmeasured robot deformation."""

import copy
import threading
import time
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from prepare_observed_stretch_fixture import prepare

from emet.controller.manipulation.encoder_capture import read_local_sim_motor_capture
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.stretch_encoder_kinematics import estimate_stretch_ee_from_encoders
from emet.robots.stretch import STRETCH_ARM_CHAIN
from emet.simulation.stretch_mujoco.datamodels.status_stretch_camera import StatusStretchCameras
from emet.simulation.stretch_mujoco.enums.stretch_cameras import StretchCameras
from emet.simulation.synchronized_capture import (
    capture_scene_for_sensors,
    capture_stretch_motors,
    robot_kinematic_signature,
)


@pytest.fixture
def encoder_setup(tmp_path):
    prepare(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "stretch.xml"))
    data = mujoco.MjData(model)
    probe, packet = capture_scene_for_sensors(
        model, data, threading.Lock(), cameras={"cam_d435i_rgb": ("d435i_camera_rgb", True)}
    )
    motor = capture_stretch_motors(model, probe, packet)
    imagery = StatusStretchCameras.default()
    for cam in (StretchCameras.cam_d435i_rgb, StretchCameras.cam_d435i_depth):
        imagery.set_camera_data(
            cam, np.ones((2, 2)), timestamp_ns=packet["interval_ns"][0], capture_end_ns=packet["interval_ns"][1]
        )
    imagery.synchronized_motor_sample = motor
    # No full-qpos debug packet is required by the genuine motor consumer.
    timing = imagery.image_timing_for(StretchCameras.cam_d435i_rgb)
    assert "robot_sample" not in timing
    obs = SimpleNamespace(
        camera_pose=np.array(packet["camera_poses"]["cam_d435i_rgb"]),
        is_simulation=True,
        image_timing={"head_cam_left": timing},
        received_monotonic=time.monotonic(),
    )
    return model, probe, packet, motor, obs


def test_motor_feedback_has_no_passive_angles_or_telescope_split(encoder_setup):
    model, probe, packet, motors, obs = encoder_setup
    measured = read_local_sim_motor_capture(obs)
    assert len(measured.names) == 8
    assert measured.as_dict() == motors["positions"]
    assert set(measured.names) == {
        "arm",
        "lift",
        "wrist_yaw",
        "wrist_pitch",
        "wrist_roll",
        "gripper",
        "head_pan",
        "head_tilt",
    }
    with pytest.raises(ValueError):
        measured.positions.setflags(write=True)
    before = copy.deepcopy(motors["positions"])
    probe.joint("rubber_left_x").qpos[0] = 0.3
    probe.joint("joint_gripper_finger_left_open").qpos[0] = 0.2
    probe.joint("joint_arm_l0").qpos[0] += 0.01
    probe.joint("joint_arm_l1").qpos[0] -= 0.01
    mujoco.mj_kinematics(model, probe)
    again = capture_stretch_motors(model, probe, packet)
    np.testing.assert_allclose(list(again["positions"].values()), list(before.values()), atol=1e-12)


@pytest.mark.parametrize(
    "failure", ["hardware", "stale", "other_clock", "passive_angle", "missing_arm", "nan", "different_depth"]
)
def test_invalid_or_fabricated_motor_packets_reject(encoder_setup, failure):
    model, probe, packet, motors, obs = encoder_setup
    timing = obs.image_timing["head_cam_left"]
    motor = timing["motor_sample"]
    if failure == "hardware":
        obs.is_simulation = False
    elif failure == "stale":
        obs.received_monotonic += 5.0
    elif failure == "other_clock":
        motor["clock_id"] = "other"
    elif failure == "passive_angle":
        motor["positions"]["rubber_left_x"] = 0.0
    elif failure == "missing_arm":
        motor["positions"].pop("arm")
    elif failure == "nan":
        motor["positions"]["arm"] = float("nan")
    elif failure == "different_depth":
        timing["depth_capture_interval_ns"][1] += 1
    with pytest.raises(ObservationUnavailable):
        read_local_sim_motor_capture(obs, now=obs.received_monotonic + 0.001)


@pytest.mark.parametrize("change", ["gear", "reference"])
def test_motor_name_cannot_hide_different_transmission_calibration(encoder_setup, change):
    model, probe, packet, motors, obs = encoder_setup
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift")
    if change == "gear":
        model.actuator_gear[aid, 0] = 2.0
    else:
        model.qpos0[model.joint("joint_lift").qposadr[0]] = 0.1
    with pytest.raises(ValueError, match="unsupported_stretch_encoder_calibration"):
        capture_stretch_motors(model, probe, packet)


def test_encoder_fk_preserves_rigid_ee_without_measuring_passive_fingers(encoder_setup):
    model, probe, packet, motors, obs = encoder_setup
    probe.joint("rubber_left_x").qpos[0] = 0.3
    probe.joint("joint_gripper_finger_left_open").qpos[0] = 0.2
    probe.joint("joint_arm_l0").qpos[0] = 0.03
    probe.joint("joint_arm_l1").qpos[0] = 0.01
    probe.joint("joint_arm_l2").qpos[0] = 0.02
    probe.joint("joint_arm_l3").qpos[0] = 0.04
    mujoco.mj_kinematics(model, probe)
    obs.image_timing["head_cam_left"]["motor_sample"] = capture_stretch_motors(model, probe, packet)
    estimated = estimate_stretch_ee_from_encoders(obs, model, now=obs.received_monotonic + 0.001)
    actual = probe.body(STRETCH_ARM_CHAIN.ee_body)
    np.testing.assert_allclose(estimated.world_from_ee[:3, 3], actual.xpos, atol=1e-9)
    np.testing.assert_allclose(estimated.world_from_ee[:3, :3], actual.xmat.reshape(3, 3), atol=1e-9)
    assert "rubber_left_x" in estimated.inferred_joint_names
    assert not estimated.gripper_geometry_qualified and not estimated.execution_qualified
    assert "robot_sample" not in obs.image_timing["head_cam_left"]


def test_camera_extrinsic_change_rejects_encoder_fk_instead_of_shifting_robot(encoder_setup):
    model, probe, packet, motors, obs = encoder_setup
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "d435i_camera_rgb")
    model.cam_pos[cid, 0] += 0.01
    with pytest.raises(ObservationUnavailable, match="robot_capture_model_mismatch"):
        estimate_stretch_ee_from_encoders(obs, model, now=obs.received_monotonic + 0.001)


def test_unknown_joint_in_rigid_ee_chain_is_rejected(encoder_setup, tmp_path):
    model, probe, packet, motors, obs = encoder_setup
    xml = ET.parse(tmp_path / "stretch.xml")
    ee = xml.getroot().find(".//body[@name='link_grasp_center']")
    ET.SubElement(ee, "joint", {"name": "unmeasured_wrist_flex", "type": "hinge"})
    ET.SubElement(ee, "inertial", {"pos": "0 0 0", "mass": ".01", "diaginertia": ".001 .001 .001"})
    path = tmp_path / "changed.xml"
    xml.write(path)
    changed = mujoco.MjModel.from_xml_path(str(path))
    obs.image_timing["head_cam_left"]["motor_sample"]["kinematic_signature"] = robot_kinematic_signature(changed)
    with pytest.raises(ObservationUnavailable, match="robot_frame_has_unmeasured_joints"):
        estimate_stretch_ee_from_encoders(obs, changed, now=obs.received_monotonic + 0.001)
