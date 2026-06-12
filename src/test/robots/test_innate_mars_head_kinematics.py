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

import numpy as np
import pytest

from emet.robots.innate_mars.head_kinematics import (
    camera_pose_in_base_link,
    compare_mjcf_camera_to_zmq,
    infer_joint_head_from_camera_pose,
)


@pytest.mark.sim
def test_sim_zmq_camera_pose_matches_mjcf_fk():
    """Innate Mars sim: ZMQ ``camera_pose`` must match local MJCF FK (OpenCV convention)."""
    pytest.importorskip("mujoco")
    import pickle
    import subprocess
    import time

    import zmq

    proc = subprocess.Popen(
        ["uv", "run", "emet", "serve", "mujoco", "--robot", "innate_mars", "--headless"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for _ in range(100):
            line = proc.stdout.readline() if proc.stdout else ""
            if "Server running" in line or "4401" in line:
                break
            time.sleep(0.2)
        time.sleep(0.8)
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.connect("tcp://127.0.0.1:4401")
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.setsockopt(zmq.RCVTIMEO, 8000)
        obs = pickle.loads(sock.recv())
        metrics = compare_mjcf_camera_to_zmq(obs)
        assert metrics["pos_err_m"] < 0.001
        assert metrics["rot_err_deg"] < 0.5
        assert obs.get("joint_head") is not None
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def test_camera_pose_in_base_link_identity():
    gps = np.zeros(2)
    compass = np.zeros(1)
    t_cam = np.eye(4)
    t_cam[0, 3] = 1.0
    out = camera_pose_in_base_link(gps, compass, t_cam)
    np.testing.assert_allclose(out[0, 3], 1.0)


def test_camera_pose_in_base_link_with_navigation_origin():
    session = {"navigation_origin_xyt": [10.0, 5.0, 0.0]}
    gps = np.zeros(2)
    compass = np.zeros(1)
    t_cam = np.eye(4)
    t_cam[0, 3] = 11.0
    out = camera_pose_in_base_link(gps, compass, t_cam, session=session)
    np.testing.assert_allclose(out[0, 3], 1.0, atol=1e-9)


def test_infer_joint_head_recovers_known_angle():
    pytest.importorskip("mujoco")
    from emet.robots.innate_mars.head_kinematics import _mjcf_camera_in_base

    joint = np.zeros(10, dtype=np.float64)
    true_ang = 0.12
    t_cam = _mjcf_camera_in_base(joint, true_ang)
    inferred = infer_joint_head_from_camera_pose(joint, t_cam, gps=np.zeros(2), compass=np.zeros(1))
    assert abs(inferred - true_ang) < 0.02


def test_patch_hardware_head_cameras():
    pytest.importorskip("mujoco")
    import mujoco

    from emet.robots.innate_mars import InnateMarsBackend
    from emet.robots.innate_mars.head_kinematics import (
        HARDWARE_HEAD_CAMERA_MOUNTS,
        HARDWARE_HEAD_VISUAL_POS,
        patch_innate_mars_head_visual_for_hardware,
        patch_innate_mars_model_for_hardware_replay,
    )

    spec = InnateMarsBackend().get_spec()
    model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
    assert patch_innate_mars_model_for_hardware_replay(model)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_left")
    np.testing.assert_allclose(model.cam_pos[cid], HARDWARE_HEAD_CAMERA_MOUNTS["head_left"]["pos"], atol=1e-9)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head_visual")
    np.testing.assert_allclose(model.body_pos[bid], HARDWARE_HEAD_VISUAL_POS, atol=1e-9)
    assert patch_innate_mars_head_visual_for_hardware(model)


def test_obs_pose_for_base_relative_mjcf_replay_zeros_planar_joints():
    from emet.robots.innate_mars.head_kinematics import obs_pose_for_base_relative_mjcf_replay

    obs = {"joint": np.array([1.0, 2.0, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6])}
    out = obs_pose_for_base_relative_mjcf_replay(obs)
    np.testing.assert_allclose(out["joint"][:3], 0.0)
    np.testing.assert_allclose(out["joint"][3:], obs["joint"][3:])


def test_hardware_visual_yaw_aligns_sim_mesh_forward_to_base_x():
    pytest.importorskip("mujoco")
    import mujoco

    from emet.robots.innate_mars import InnateMarsBackend
    from emet.robots.innate_mars.head_kinematics import (
        HARDWARE_MJCF_VISUAL_YAW_RAD,
        patch_innate_mars_head_cameras_for_hardware,
    )
    from emet.simulation.mujoco_gt_objects import camera_pose_world_opencv
    from emet.visualization.mjcf_rerun_robot import _apply_base_yaw_fix_to_points, apply_zmq_obs_to_mujoco_data

    spec = InnateMarsBackend().get_spec()
    model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
    data = mujoco.MjData(model)
    obs = {"gps": np.zeros(2), "compass": np.zeros(1), "joint": np.zeros(10)}
    apply_zmq_obs_to_mujoco_data(
        model,
        data,
        obs,
        joint_names=tuple(spec.joint_names),
        dof=spec.dof,
        base_link_name=spec.base_link_name,
        nav_origin_slot=[None],
        free_qadr=None,
    )
    mujoco.mj_forward(model, data)
    sim_gaze = camera_pose_world_opencv(model, data, "head_left")[:3, :3] @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(sim_gaze, np.array([0.0, -1.0, 0.0]), atol=1e-3)

    patch_innate_mars_head_cameras_for_hardware(model)
    mujoco.mj_forward(model, data)
    hw_gaze = camera_pose_world_opencv(model, data, "head_left")[:3, :3] @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(hw_gaze, np.array([1.0, 0.0, 0.0]), atol=1e-3)

    sim_forward = np.array([0.0, -1.0, 0.0])
    fixed = _apply_base_yaw_fix_to_points(sim_forward.reshape(1, 3), HARDWARE_MJCF_VISUAL_YAW_RAD)[0]
    np.testing.assert_allclose(fixed, np.array([1.0, 0.0, 0.0]), atol=1e-3)


def test_enrich_obs_pose_joint_head_uses_camera_fk_on_hardware():
    pytest.importorskip("mujoco")
    import mujoco

    from emet.core.zmq_protocol import EMET_ZMQ_SESSION_KEY
    from emet.robots.innate_mars import InnateMarsBackend
    from emet.robots.innate_mars.head_kinematics import (
        enrich_obs_pose_joint_head_for_hardware_replay,
        patch_innate_mars_model_for_hardware_replay,
    )

    spec = InnateMarsBackend().get_spec()
    model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
    patch_innate_mars_model_for_hardware_replay(model)
    obs = {
        EMET_ZMQ_SESSION_KEY: {"is_simulation": False},
        "gps": np.zeros(2),
        "compass": np.zeros(1),
        "joint": np.zeros(10),
        "joint_head": 0.99,
        "camera_pose": np.eye(4),
    }
    out = enrich_obs_pose_joint_head_for_hardware_replay(model, obs)
    assert abs(float(out["joint_head"]) - 0.99) > 0.1


def test_is_hardware_innate_mars_obs():
    from emet.core.zmq_protocol import EMET_ZMQ_SESSION_KEY
    from emet.robots.innate_mars.head_kinematics import is_hardware_innate_mars_obs

    assert is_hardware_innate_mars_obs({EMET_ZMQ_SESSION_KEY: {"is_simulation": False}})
    assert not is_hardware_innate_mars_obs({EMET_ZMQ_SESSION_KEY: {"is_simulation": True}})
