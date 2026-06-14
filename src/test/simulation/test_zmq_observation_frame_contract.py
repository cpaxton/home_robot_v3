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
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from emet.core.zmq_protocol import EMET_ZMQ_ROBOT_ID_KEY, EMET_ZMQ_SESSION_KEY
from emet.simulation.mujoco_server_stretch import MujocoZmqServer
from emet.simulation.zmq_frame_contract import (
    assert_camera_pose_is_mujoco_world,
    assert_zmq_observation_frames_consistent,
    episode_relative_camera_pose,
)
from emet.utils.geometry import pose_global_to_base

_ORIGIN = np.array([4.0, -2.0, 0.3], dtype=np.float64)
_WORLD_CAM = np.eye(4, dtype=np.float64)
_WORLD_CAM[:3, 3] = [4.1, -1.9, 1.2]


def _minimal_stretch_server() -> MujocoZmqServer:
    server = MujocoZmqServer.__new__(MujocoZmqServer)
    server._initial_xyt = _ORIGIN.copy()
    server._last_step = 0
    server.recv_address = "tcp://127.0.0.1:1"
    server.head_K = np.eye(3)
    server._emet_session = {EMET_ZMQ_SESSION_KEY: {"navigation_origin_xyt": _ORIGIN.tolist()}}
    server._camera_data = MagicMock(
        cam_d435i_rgb=np.zeros((64, 64, 3), dtype=np.uint8), cam_d435i_depth=np.ones((64, 64))
    )
    server.robot_sim = MagicMock()
    server.robot_sim.get_ee_pose.return_value = np.eye(4)
    return server


def test_stretch_full_obs_camera_pose_is_world_not_episode_relative():
    server = _minimal_stretch_server()
    with (
        patch.object(server, "_stretch_sim_publish_ok", return_value=True),
        patch.object(server, "get_base_pose", return_value=np.zeros(3)),
        patch.object(server, "get_joint_state", return_value=(np.zeros(11), np.zeros(11), np.zeros(11))),
        patch.object(server, "get_control_mode", return_value="navigation"),
        patch.object(server, "base_controller_at_goal", return_value=True),
        patch.object(server, "_head_camera_opencv_world", return_value=_WORLD_CAM.copy()),
        patch("emet.simulation.mujoco_server_stretch.compression.to_jpg", side_effect=lambda x: x),
        patch("emet.simulation.mujoco_server_stretch.compression.to_jp2", side_effect=lambda x: x),
    ):
        msg = server.get_full_observation_message()
    assert msg is not None
    np.testing.assert_allclose(msg["camera_pose"][:3, 3], _WORLD_CAM[:3, 3], atol=1e-6)
    ep = pose_global_to_base(_WORLD_CAM, _ORIGIN)
    assert float(np.linalg.norm(msg["camera_pose"][:3, 3] - ep[:3, 3])) > 0.5


def test_stretch_servo_head_cam_pose_is_world():
    server = _minimal_stretch_server()
    server.ee_K = np.eye(3)
    server.image_scaling = 1.0
    server.ee_image_scaling = 1.0
    server.ee_depth_scaling = 1.0
    server.depth_scaling = 1.0
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    dep = np.ones((64, 64), dtype=np.float32)
    server._camera_data = MagicMock(cam_d435i_rgb=img, cam_d435i_depth=dep, cam_d405_rgb=img, cam_d405_depth=dep)
    ee_cam = np.eye(4)
    ee_cam[:3, 3] = [4.2, -1.8, 1.0]
    with (
        patch.object(server, "_stretch_sim_publish_ok", return_value=True),
        patch.object(server, "get_joint_state", return_value=(np.zeros(11), np.zeros(11), np.zeros(11))),
        patch.object(server, "_head_camera_opencv_world", return_value=_WORLD_CAM.copy()),
        patch.object(server.robot_sim, "get_link_pose", return_value=ee_cam),
        patch.object(server.robot_sim, "get_ee_pose", return_value=np.eye(4)),
        patch.object(server, "_rescale_color_and_depth", side_effect=lambda c, d, s: (c, d)),
        patch("emet.simulation.mujoco_server_stretch.compression.to_jpg", side_effect=lambda x: x),
        patch("emet.simulation.mujoco_server_stretch.compression.to_jp2", side_effect=lambda x: x),
    ):
        msg = server.get_servo_message()
    assert msg is not None
    np.testing.assert_allclose(msg["head_cam/pose"][:3, 3], _WORLD_CAM[:3, 3], atol=1e-6)
    np.testing.assert_allclose(msg["ee_cam/pose"][:3, 3], ee_cam[:3, 3], atol=1e-6)


def test_stretch_episode_relative_helper_differs_from_published_pose():
    server = _minimal_stretch_server()
    with (
        patch.object(server, "_stretch_sim_publish_ok", return_value=True),
        patch.object(server, "get_base_pose", return_value=np.zeros(3)),
        patch.object(server, "get_joint_state", return_value=(np.zeros(11), np.zeros(11), np.zeros(11))),
        patch.object(server, "get_control_mode", return_value="navigation"),
        patch.object(server, "base_controller_at_goal", return_value=True),
        patch.object(server, "_head_camera_opencv_world", return_value=_WORLD_CAM.copy()),
        patch.object(server, "get_head_camera_pose", return_value=episode_relative_camera_pose(_WORLD_CAM, _ORIGIN)),
        patch("emet.simulation.mujoco_server_stretch.compression.to_jpg", side_effect=lambda x: x),
        patch("emet.simulation.mujoco_server_stretch.compression.to_jp2", side_effect=lambda x: x),
    ):
        msg = server.get_full_observation_message()
        ep = server.get_head_camera_pose()
    assert msg is not None and ep is not None
    assert not np.allclose(msg["camera_pose"][:3, 3], ep[:3, 3], atol=1e-3)


def test_assert_zmq_observation_frames_consistent_rejects_episode_relative_camera():
    bad = {
        "gps": np.array([0.0, 0.0]),
        "compass": np.array([0.0]),
        "camera_pose": np.eye(4),
        EMET_ZMQ_SESSION_KEY: {"navigation_origin_xyt": [5.0, 5.0, 0.0]},
        EMET_ZMQ_ROBOT_ID_KEY: "stretch",
    }
    with pytest.raises(AssertionError, match="episode-relative"):
        assert_zmq_observation_frames_consistent(bad)


def test_assert_zmq_observation_frames_consistent_accepts_good_obs():
    origin = np.array([3.0, -1.0, 0.0])
    cam = np.eye(4)
    cam[:3, 3] = [3.05, -0.95, 1.1]
    good = {
        "gps": np.array([0.0, 0.0]),
        "compass": np.array([0.0]),
        "camera_pose": cam,
        EMET_ZMQ_SESSION_KEY: {"navigation_origin_xyt": origin.tolist()},
    }
    assert_zmq_observation_frames_consistent(good)


def test_assert_camera_pose_is_mujoco_world_spawn_check():
    origin = np.array([2.0, 3.0, 0.0])
    world_cam = np.eye(4)
    world_cam[:3, 3] = [2.1, 3.05, 1.0]
    assert_camera_pose_is_mujoco_world(
        world_cam,
        navigation_origin_xyt=origin,
        gps=[0.0, 0.0],
        compass=[0.0],
    )
    ep_cam = episode_relative_camera_pose(world_cam, origin)
    with pytest.raises(AssertionError):
        assert_camera_pose_is_mujoco_world(
            ep_cam,
            navigation_origin_xyt=origin,
            gps=[0.0, 0.0],
            compass=[0.0],
        )


def test_habitat_sqa3d_style_obs_without_navigation_origin_passes_contract():
    """Habitat / SQA3D: gps and camera_pose are both absolute world (no episode origin split)."""
    # Habitat-style: gps is (x, z) world; camera near agent head in same frame.
    cam = np.eye(4, dtype=np.float64)
    cam[:3, 3] = [1.05, 1.52, 0.98]
    msg = {
        "gps": np.array([1.0, 2.0]),
        "compass": np.array([0.1]),
        "camera_pose": cam,
        EMET_ZMQ_SESSION_KEY: {"sim_object_placements": {}},
    }
    assert_zmq_observation_frames_consistent(msg)
