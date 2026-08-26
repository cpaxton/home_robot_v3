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

import os
from unittest.mock import patch

import numpy as np
import pytest

from emet.core.zmq_protocol import EMET_ZMQ_SESSION_KEY
from emet.simulation.robosuite_load_utils import update_robot_qpos0_from_data
from emet.simulation.sim_object_placements import apply_navigation_origin_to_session
from emet.simulation.zmq_frame_contract import assert_zmq_observation_frames_consistent

_run_sim = os.environ.get("RUN_SIM_TESTS", "1").strip().lower()
RUN_SIM_TESTS = _run_sim not in ("0", "false", "no", "off")


def _innate_mars_pole_server():
    pytest.importorskip("mujoco")
    import mujoco

    from emet.robots.innate_mars import InnateMarsBackend
    from emet.simulation.circle_calibration import build_merged_model_with_pole_ring
    from emet.simulation.mujoco_server import _load_default_scene_with_robot
    from emet.simulation.robosuite_server import RobosuiteZmqServer

    base = _load_default_scene_with_robot("innate_mars")
    if base is None:
        pytest.skip("Merged innate_mars scene not available")
    data0 = mujoco.MjData(base)
    mujoco.mj_forward(base, data0)
    bid = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if bid < 0:
        pytest.skip("base_link missing")
    cx0, cy0 = float(data0.body(bid).xpos[0]), float(data0.body(bid).xpos[1])
    model = build_merged_model_with_pole_ring(cx=cx0, cy=cy0)
    spec = InnateMarsBackend().get_spec()
    server = RobosuiteZmqServer(
        robot_spec=spec,
        scene_model=model,
        send_port=0,
        recv_port=0,
        send_state_port=0,
        send_servo_port=0,
        use_remote_computer=False,
    )
    return server


def _finish_robocasa_session_startup(server, *, robocasa: bool = False) -> None:
    """Mimic fixed ``start()`` tail: session → post-GT reapply → navigation_origin."""
    spawn_floor_map = server._compute_robocasa_spawn_floor_map() if robocasa else None
    server._nav_world_clip_rect = None
    if spawn_floor_map is not None:
        eroded = spawn_floor_map.get("clip_eroded_xy")
        if isinstance(eroded, (list, tuple)) and len(eroded) == 4:
            server._nav_world_clip_rect = tuple(float(v) for v in eroded)
    server._emet_session = server._build_emet_session(robocasa=robocasa, spawn_floor_map=spawn_floor_map)
    if server._planar_autoplace_snap_qpos0:
        with server._mj_lock:
            if not server._reapply_planar_autoplace_world_xyt():
                server._restore_planar_base_from_qpos0()
            update_robot_qpos0_from_data(server._mjmodel, server._mjdata, server._spec)
            server._preserve_joint_ctrl_hold_from_ctrl()
    server._initial_xyt = server.get_base_xyt()
    if server._emet_session is not None and server._initial_xyt is not None:
        apply_navigation_origin_to_session(server._emet_session, server._initial_xyt)


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
def test_robosuite_navigation_origin_matches_base_after_session_build():
    server = _innate_mars_pole_server()
    server._load_model()
    server._stabilize_physics_state_after_load()
    if server._planar_autoplace_snap_qpos0:
        with server._mj_lock:
            if not server._reapply_planar_autoplace_world_xyt():
                server._restore_planar_base_from_qpos0()
    _finish_robocasa_session_startup(server, robocasa=False)

    origin = np.asarray(server._emet_session["navigation_origin_xyt"], dtype=np.float64)
    base_xyt = server.get_base_xyt()
    gps = server.get_base_pose()
    assert gps is not None
    np.testing.assert_allclose(origin[:3], base_xyt[:3], atol=0.02)
    assert float(np.linalg.norm(gps[:2])) < 0.05
    assert abs(float(gps[2])) < 0.08


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
def test_robosuite_full_obs_camera_pose_is_world():
    server = _innate_mars_pole_server()
    server._load_model()
    server._stabilize_physics_state_after_load()
    _finish_robocasa_session_startup(server, robocasa=False)
    server._running = True

    primary = server._spec.camera_names[0]
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    depth = np.ones((120, 160), dtype=np.float32) * 0.5
    k = np.eye(3)
    with patch.object(server, "_primary_rgb_and_depth", return_value=(rgb, depth, k)):
        msg = server.get_full_observation_message()
    assert msg is not None
    assert msg.get("lidar_points") is not None
    assert msg["lidar_points"].ndim == 2 and msg["lidar_points"].shape[1] == 2
    assert msg.get("lidar_timestamp") is not None
    expected = server._camera_pose_world(primary)
    np.testing.assert_allclose(msg["camera_pose"], expected, atol=1e-4)


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
def test_robosuite_obs_passes_frame_contract_at_spawn():
    server = _innate_mars_pole_server()
    server._load_model()
    server._stabilize_physics_state_after_load()
    _finish_robocasa_session_startup(server, robocasa=False)
    server._running = True

    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    depth = np.ones((120, 160), dtype=np.float32) * 0.5
    k = np.eye(3)
    with patch.object(server, "_primary_rgb_and_depth", return_value=(rgb, depth, k)):
        msg = server.get_full_observation_message()
    assert msg is not None
    assert EMET_ZMQ_SESSION_KEY in msg
    assert_zmq_observation_frames_consistent(msg)
