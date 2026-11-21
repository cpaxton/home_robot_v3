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

from emet.controller.manipulation.encoder_capture import read_local_sim_motor_capture, read_local_sim_wheel_capture
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.stretch_encoder_kinematics import (
    estimate_stretch_ee_from_encoders,
    stretch_base_from_head_camera,
)
from emet.mapping.wheel_odometry import DifferentialDriveOdometry
from emet.robots.stretch import STRETCH_ARM_CHAIN
from emet.simulation.stretch_mujoco.datamodels.status_stretch_camera import StatusStretchCameras
from emet.simulation.stretch_mujoco.enums.stretch_cameras import StretchCameras
from emet.simulation.synchronized_capture import (
    capture_scene_for_sensors,
    capture_stretch_motors,
    capture_stretch_wheels,
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


def test_wheel_odometry_uses_only_ordered_same_capture_wheel_angles(encoder_setup):
    model, probe, _, _, _ = encoder_setup
    odometry = DifferentialDriveOdometry(wheel_radius_m=.05, track_width_m=.3)

    def observation(left, right):
        probe.joint("joint_left_wheel").qpos[0] = left
        probe.joint("joint_right_wheel").qpos[0] = right
        sample, packet = capture_scene_for_sensors(model, probe, threading.Lock(), cameras={})
        probe.joint("joint_right_wheel").qpos[0] += 2.
        imagery = StatusStretchCameras.default()
        for cam in (StretchCameras.cam_d435i_rgb, StretchCameras.cam_d435i_depth):
            imagery.set_camera_data(
                cam, np.ones((2, 2)), timestamp_ns=packet["interval_ns"][0], capture_end_ns=packet["interval_ns"][1]
            )
        imagery.synchronized_wheel_sample = capture_stretch_wheels(model, sample, packet)
        timing = imagery.image_timing_for(StretchCameras.cam_d435i_rgb)
        assert "robot_sample" not in timing and "motor_sample" not in timing
        return SimpleNamespace(
            is_simulation=True, image_timing={"head_cam_left": timing}, received_monotonic=time.monotonic()
        )

    # Nonzero absolute counters are zeroed in the first frame.
    first = read_local_sim_wheel_capture(observation(15., 17.))
    straight = read_local_sim_wheel_capture(observation(25., 27.))
    np.testing.assert_allclose(odometry.update(first), [0., 0., 0.])
    np.testing.assert_allclose(odometry.update(straight), [.5, 0., 0.])
    with pytest.raises(ObservationUnavailable, match="wheel_odometry_capture_out_of_order"):
        odometry.update(first)

    # A quarter circle with radius .15 m follows from pivoting around the left wheel.
    arc_obs = observation(25., 27. + 3 * np.pi)
    bad_obs = copy.deepcopy(arc_obs)
    bad_obs.image_timing["head_cam_left"]["depth_capture_interval_ns"][1] += 1
    with pytest.raises(ObservationUnavailable, match="capture_motion_alignment_unknown"):
        read_local_sim_wheel_capture(bad_obs)
    arc = read_local_sim_wheel_capture(arc_obs)
    np.testing.assert_allclose(odometry.update(arc), [.65, .15, np.pi / 2], atol=1e-12)


def test_encoder_navigation_server_needs_fresh_synchronized_wheel_and_head_capture(encoder_setup, monkeypatch):
    from emet.eval import stretch_encoder_server

    model, probe, packet, _, obs = encoder_setup
    server = stretch_encoder_server.StretchEncoderNavigationServer.__new__(
        stretch_encoder_server.StretchEncoderNavigationServer
    )
    server._encoder_model = model
    server._wheel_odometry = DifferentialDriveOdometry(wheel_radius_m=.05, track_width_m=.3)
    server._encoder_lock = threading.Lock()
    server._encoder_frame = None
    server._motion_guard = None
    server.active = False
    server._base_controller_at_goal = False
    timing = obs.image_timing["head_cam_left"]
    timing["robot_sample"] = packet
    server._camera_data = SimpleNamespace(image_timing_for=lambda _: copy.deepcopy(timing))
    now = [obs.received_monotonic]
    monkeypatch.setattr(stretch_encoder_server, "time", SimpleNamespace(monotonic=lambda: now[0]))
    assert server.get_base_pose() is None
    assert server._full_observation_poses(server._camera_data) is None

    timing["wheel_sample"] = capture_stretch_wheels(model, probe, packet)
    xyt, camera, ee = server._full_observation_poses(server._camera_data)
    np.testing.assert_array_equal(server.get_base_pose(), xyt)
    np.testing.assert_allclose(camera, stretch_base_from_head_camera(read_local_sim_motor_capture(obs), model))
    assert ee is None
    assert server.navigation_policy_measurement()["timestamp"] == timing["timestamp_ns"] / 1e9

    now[0] += 1.001
    assert server.get_base_pose() is None
    assert server.navigation_policy_measurement() == {"pose": None, "timestamp": None, "stopped": False}
    with pytest.raises(RuntimeError, match="fresh wheel feedback"):
        server.start_navigation_command({"xyt": [0.5, 0., 0.]})


def test_changed_telescope_transmission_cannot_reuse_a_motor_capture(encoder_setup):
    model, probe, packet, motors, obs = encoder_setup
    tendon = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "extend")
    model.wrap_prm[int(model.tendon_adr[tendon])] = 2.0
    with pytest.raises(ObservationUnavailable, match="robot_capture_model_mismatch"):
        estimate_stretch_ee_from_encoders(obs, model)


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


def test_encoder_head_pose_is_base_relative_without_a_localized_camera_pose(encoder_setup):
    model, probe, packet, _, obs = encoder_setup
    probe.qpos[:7] = [.4, -.3, .2, np.cos(.15), 0., 0., np.sin(.15)]
    probe.joint("joint_head_pan").qpos[0] = .27
    probe.joint("joint_head_tilt").qpos[0] = -.9
    _, fresh = capture_scene_for_sensors(
        model, probe, threading.Lock(), cameras={"cam_d435i_rgb": ("d435i_camera_rgb", True)}
    )
    # Keep the timing/encoders together and provide no camera pose to the consumer.
    timing = obs.image_timing["head_cam_left"]
    timing["timestamp_ns"] = fresh["interval_ns"][0]
    timing["capture_interval_ns"] = timing["depth_capture_interval_ns"] = fresh["interval_ns"]
    timing["motor_sample"] = capture_stretch_motors(model, probe, fresh)
    del obs.camera_pose
    obs.received_monotonic = time.monotonic()
    measured = read_local_sim_motor_capture(obs, now=obs.received_monotonic)
    base_from_head = stretch_base_from_head_camera(measured, model)

    mujoco.mj_kinematics(model, probe)
    base = probe.body("base_link")
    world_from_base = np.eye(4)
    world_from_base[:3, :3] = base.xmat.reshape(3, 3)
    world_from_base[:3, 3] = base.xpos
    np.testing.assert_allclose(world_from_base @ base_from_head, fresh["camera_poses"]["cam_d435i_rgb"], atol=1e-9)


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
