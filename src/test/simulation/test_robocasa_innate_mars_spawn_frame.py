# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Robocasa kitchen + innate_mars: autoplace spawn must match ZMQ nav/camera frames."""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest

_run_sim = os.environ.get("RUN_SIM_TESTS", "1").strip().lower()
RUN_SIM_TESTS = _run_sim not in ("0", "false", "no", "off")


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
@pytest.mark.timeout(180)
def test_robocasa_innate_mars_spawn_inside_walkable_clip_and_frame_contract():
    pytest.importorskip("mujoco")
    from emet.robots.innate_mars import InnateMarsBackend
    from emet.simulation.robosuite_server import RobosuiteZmqServer
    from emet.simulation.sim_object_placements import apply_navigation_origin_to_session
    from emet.simulation.stretch_mujoco.robocasa_gen import model_generation_wizard
    from emet.simulation.zmq_frame_contract import assert_zmq_observation_frames_consistent

    model, _xml, objects_info = model_generation_wizard(
        task="PickPlaceCounterToCabinet", layout=1, style=1, robot="innate_mars"
    )
    spawn_hint = np.asarray(objects_info["_emet_spawn_hint_xyt"], dtype=np.float64).reshape(-1)[:3]
    spec = InnateMarsBackend().get_spec()
    server = RobosuiteZmqServer(
        robot_spec=spec,
        scene_model=model,
        send_port=0,
        recv_port=0,
        send_state_port=0,
        send_servo_port=0,
        environment={"kind": "robocasa", "spawn_hint_xyt": spawn_hint.tolist()},
    )
    server._load_model()
    server._stabilize_physics_state_after_load()
    if server._planar_autoplace_snap_qpos0:
        with server._mj_lock:
            assert server._reapply_planar_autoplace_world_xyt()
    server._initial_xyt = server.get_base_xyt()
    spawn_floor_map = server._compute_robocasa_spawn_floor_map()
    server._nav_world_clip_rect = None
    if spawn_floor_map is not None:
        eroded = spawn_floor_map.get("clip_eroded_xy")
        if isinstance(eroded, (list, tuple)) and len(eroded) == 4:
            server._nav_world_clip_rect = tuple(float(v) for v in eroded)
    server._emet_session = server._build_emet_session(robocasa=True, spawn_floor_map=spawn_floor_map)
    apply_navigation_origin_to_session(server._emet_session, server._initial_xyt)
    server._running = True

    clip = server._nav_world_clip_rect
    assert clip is not None, "expected Robocasa walkable clip"
    x0, x1, y0, y1 = clip
    bx, by = float(server._initial_xyt[0]), float(server._initial_xyt[1])
    assert x0 <= bx <= x1 and y0 <= by <= y1, (
        f"spawn ({bx:.3f}, {by:.3f}) outside walkable clip x=[{x0:.3f},{x1:.3f}] y=[{y0:.3f},{y1:.3f}]"
    )
    hint_dxy = float(np.hypot(bx - float(spawn_hint[0]), by - float(spawn_hint[1])))
    assert hint_dxy < 0.15, (
        f"spawn ({bx:.3f}, {by:.3f}) moved {hint_dxy:.3f}m from robosuite hint "
        f"({float(spawn_hint[0]):.3f}, {float(spawn_hint[1]):.3f}); "
        "planar autoplace must respect init_robot_base_pos like Stretch"
    )

    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    depth = np.ones((120, 160), dtype=np.float32) * 0.5
    k = np.eye(3)
    with patch.object(server, "_primary_rgb_and_depth", return_value=(rgb, depth, k)):
        msg = server.get_full_observation_message()
    assert msg is not None
    assert_zmq_observation_frames_consistent(msg)
