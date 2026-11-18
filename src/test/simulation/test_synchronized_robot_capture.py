# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Co-sampled sensor data, robot-only isolation, and consumer rejection checks."""

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import mujoco
import numpy as np
import pytest

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.sim_robot_capture import read_synchronized_robot_capture
from emet.simulation.stretch_mujoco.datamodels.status_stretch_camera import StatusStretchCameras
from emet.simulation.stretch_mujoco.enums.stretch_cameras import StretchCameras
from emet.simulation.stretch_mujoco.mujoco_server_camera_manager import MujocoServerCameraManagerSync
from emet.simulation.synchronized_capture import capture_scene_for_sensors, robot_kinematic_signature

ROBOT = """<body name="base_link"><freejoint/><geom type="box" size=".1 .1 .1"/>
<camera name="d435i_camera_rgb" pos=".2 0 0"/>
<camera name="d435i_camera_depth" pos=".2 0 0"/>
<body name="ee" pos="0 0 .3"><joint name="arm" type="slide" axis="0 0 1"/>
<geom type="box" size=".02 .02 .02"/></body></body>"""
OBJECT = '<body name="hidden_object" pos="2 0 0"><freejoint name="secret_object_joint"/><geom size=".1"/></body>'


def model(scene):
    return mujoco.MjModel.from_xml_string("<mujoco><worldbody>" + scene + "</worldbody></mujoco>")


def sampled():
    scene, robot = model(OBJECT + ROBOT), model(ROBOT)
    data = mujoco.MjData(scene)
    data.joint("arm").qpos[0] = 0.15
    mujoco.mj_forward(scene, data)
    probe, packet = capture_scene_for_sensors(
        scene,
        data,
        threading.Lock(),
        cameras={"cam_d435i_rgb": ("d435i_camera_rgb", True), "cam_d435i_depth": ("d435i_camera_depth", True)},
    )
    imagery = StatusStretchCameras.default()
    start, end = packet["interval_ns"]
    for camera in (StretchCameras.cam_d435i_rgb, StretchCameras.cam_d435i_depth):
        imagery.set_camera_data(camera, np.ones((2, 2)), timestamp_ns=start, capture_end_ns=end)
    imagery.synchronized_robot_sample = json.loads(json.dumps(packet))
    imagery.synchronized_camera_poses = packet["camera_poses"]
    obs = SimpleNamespace(
        is_simulation=True,
        image_timing={"head_cam_left": imagery.image_timing_for(StretchCameras.cam_d435i_rgb)},
        camera_pose=imagery.captured_camera_pose(StretchCameras.cam_d435i_rgb),
        received_monotonic=time.monotonic(),
    )
    return scene, robot, data, probe, packet, imagery, obs


def read(obs, robot):
    return read_synchronized_robot_capture(
        obs, robot, ee_body="ee", gripper_bodies=("ee",), now=obs.received_monotonic + 0.001
    )


def test_robot_only_sample_matches_robot_model_without_environment_state():
    scene, robot, data, probe, packet, imagery, obs = sampled()
    assert scene.nq > robot.nq
    assert packet["kinematic_signature"] == robot_kinematic_signature(robot)
    assert set(packet["joint_positions"]) == {"@base_link:0", "arm"}
    assert "secret_object" not in json.dumps(packet)
    before = data.qpos.copy(), data.qacc.copy(), data.qacc_warmstart.copy()
    result = read(obs, robot)
    np.testing.assert_allclose(result.robot_after.world_from_ee[:3, 3], [0, 0, 0.45])
    assert result.alignment_error_m == result.alignment_error_rad == 0.0
    assert result.source == "simulator_robot_kinematics_debug"
    for got, expected in zip((data.qpos, data.qacc, data.qacc_warmstart), before, strict=True):
        np.testing.assert_array_equal(got, expected)
    # The renderer owns its snapshot; later physics cannot change sampled pixels/FK.
    data.joint("arm").qpos[0] = 0.7
    mujoco.mj_forward(scene, data)
    assert probe.joint("arm").qpos[0] == 0.15
    assert imagery.captured_camera_pose(StretchCameras.cam_d435i_rgb) is not None


@pytest.mark.parametrize(
    "failure",
    [
        "hardware",
        "remote_clock",
        "async",
        "wrong_camera_pose",
        "missing_joint",
        "bad_joint",
        "empty_interval",
        "different_depth",
        "extra_scene_state",
        "stale",
        "wrong_model",
    ],
)
def test_unqualified_or_incoherent_samples_reject(failure):
    scene, robot, data, probe, packet, imagery, obs = sampled()
    timing = obs.image_timing["head_cam_left"]
    sample = timing["robot_sample"]
    if failure == "hardware":
        obs.is_simulation = False
    elif failure == "remote_clock":
        timing["clock_id"] = "other"
    elif failure == "async":
        timing["source"] = "renderer_scene_capture"
    elif failure == "wrong_camera_pose":
        obs.camera_pose[:3, 3] += 0.1
    elif failure == "missing_joint":
        sample["joint_positions"].pop("arm")
    elif failure == "bad_joint":
        sample["joint_positions"]["arm"] = ["not-an-encoder"]
    elif failure == "empty_interval":
        sample["interval_ns"] = timing["capture_interval_ns"] = timing["depth_capture_interval_ns"] = []
    elif failure == "different_depth":
        timing["depth_capture_interval_ns"][1] += 1
    elif failure == "extra_scene_state":
        sample["scene_qpos"] = data.qpos.tolist()
    elif failure == "stale":
        obs.received_monotonic += 5.0
    elif failure == "wrong_model":
        robot.body_pos[robot.body("ee").id, 2] += 0.1
    with pytest.raises(ObservationUnavailable):
        read(obs, robot)


def test_metadata_roundtrip_copies_packet_and_untimed_pixels_lose_alignment():
    *_, imagery, obs = sampled()
    restored = StatusStretchCameras.from_dict(imagery.to_dict())
    timing = restored.image_timing_for(StretchCameras.cam_d435i_rgb)
    timing["robot_sample"]["joint_positions"]["arm"][0] = 99.0
    assert restored.synchronized_robot_sample["joint_positions"]["arm"] == [0.15]
    restored.set_camera_data(StretchCameras.cam_d435i_rgb, np.zeros((2, 2)))
    assert restored.image_timing_for(StretchCameras.cam_d435i_rgb) is None


def test_manager_renders_all_pixels_from_private_copy_while_physics_can_advance():
    scene, robot, data, probe, packet, imagery, obs = sampled()
    manager = MujocoServerCameraManagerSync.__new__(MujocoServerCameraManagerSync)
    manager.mujoco_server = SimpleNamespace(mjmodel=scene, mjdata=data, data_proxies=MagicMock())
    manager.camera_lock = threading.Lock()
    manager.camera_fps_counter = SimpleNamespace(fps=20)
    samples = []
    manager.camera_renderers = {}
    for camera in (StretchCameras.cam_d435i_rgb, StretchCameras.cam_d435i_depth):
        renderer = MagicMock()

        def scene_update(*, data, camera):
            samples.append(float(data.joint("arm").qpos[0]))
            manager.mujoco_server.mjdata.joint("arm").qpos[0] += 0.1

        renderer.update_scene.side_effect = scene_update
        renderer.render.return_value = np.ones((2, 2))
        manager.camera_renderers[camera] = renderer
    manager.get_camera_params = lambda camera: np.eye(3)
    manager._pull_synchronized_camera_data()
    assert samples == [0.15, 0.15]
    assert data.joint("arm").qpos[0] == pytest.approx(0.35)
    captured = manager.mujoco_server.data_proxies.set_cameras.call_args.args[0]
    timing = captured.image_timing_for(StretchCameras.cam_d435i_rgb)
    assert timing["source"] == "renderer_synchronized_snapshot"
    assert timing["capture_interval_ns"] == timing["depth_capture_interval_ns"]
