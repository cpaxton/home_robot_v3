# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Robocasa planar spawn: Robosuite init_robot_base_pos hint vs walkable-clip centroid."""

from __future__ import annotations

import os
import tempfile
from threading import Lock
from unittest.mock import patch

import numpy as np
import pytest

from emet.core.zmq_protocol import EMET_ZMQ_SESSION_KEY

_run_sim = os.environ.get("RUN_SIM_TESTS", "1").strip().lower()
RUN_SIM_TESTS = _run_sim not in ("0", "false", "no", "off")


def _load_innate_mars_robocasa_kitchen():
    """One kitchen + innate_mars merge (same entry as emet serve mujoco --use-robocasa --robot innate_mars)."""
    pytest.importorskip("mujoco")
    import mujoco

    from emet.robots.innate_mars import InnateMarsBackend
    from emet.simulation import scene_base_spawn
    from emet.simulation.stretch_mujoco.robocasa_gen import model_generation_wizard

    model, xml_str, objects_info = model_generation_wizard(
        task="PickPlaceCounterToCabinet",
        layout=1,
        style=1,
        robot="innate_mars",
    )
    hint = np.asarray(objects_info["_emet_spawn_hint_xyt"], dtype=np.float64).reshape(-1)[:3]
    spec = InnateMarsBackend().get_spec()
    joint_names = tuple(str(n) for n in spec.planar_base_joint_names)
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(xml_str)
        mjcf_path = f.name

    margin = 0.85
    clip_pad = 0.47
    data_probe = mujoco.MjData(model)
    mujoco.mj_forward(model, data_probe)
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    robot_bodies = scene_base_spawn._bodies_descending_from(model, base_bid)
    xy_clip_scene = scene_base_spawn.collision_scene_xy_clip_rect(
        model,
        data_probe,
        robot_bodies=robot_bodies,
        floor_geom_name="floor",
        margin=0.55,
        max_geom_rbound=15.0,
        suppress_exterior_filter=False,
    )
    inset = float(margin) + 0.10
    xy_clip = scene_base_spawn._erode_xy_rect(xy_clip_scene, inset) if xy_clip_scene is not None else None
    if xy_clip is None:
        xy_clip = xy_clip_scene
    assert xy_clip is not None, "expected walkable clip for Robocasa kitchen"
    clip_centroid = np.array(
        [0.5 * (xy_clip[0] + xy_clip[1]), 0.5 * (xy_clip[2] + xy_clip[3])],
        dtype=np.float64,
    )

    find_kw = {
        "model": model,
        "base_body_name": "base_link",
        "joint_names": joint_names,
        "spawn_profile": "robocasa",
        "merged_mjcf_path": mjcf_path,
        "environment": {"kind": "robocasa"},
        "footprint_xy_margin_m": margin,
        "clip_edge_pad_m": clip_pad,
        "clip_guard_body_names": spec.planar_spawn_clip_guard_body_names,
        "clip_guard_pad_m": 0.4,
        "robocasa_first_clearance_m": spec.planar_spawn_robocasa_first_clearance_m,
    }
    return model, hint, clip_centroid, find_kw


def test_wait_for_obs_waits_for_navigation_origin_on_robosuite():
    """Robosuite clients must not map until navigation_origin_xyt is published."""
    from emet.controller.generic_zmq_client import GenericZmqClient

    client = GenericZmqClient.__new__(GenericZmqClient)
    client._obs_lock = Lock()
    client._obs = {"joint": [0.0] * 4}
    client._servo = None
    client._emet_session_cache = {"runtime_kind": "robosuite_sim"}
    assert client.wait_for_obs(timeout=0.15) is False

    client._emet_session_cache = {
        "runtime_kind": "robosuite_sim",
        "navigation_origin_xyt": [2.5, -1.0, 1.57],
    }
    assert client.wait_for_obs(timeout=0.15) is True

    client._emet_session_cache = {"runtime_kind": "stretch_mujoco"}
    assert client.wait_for_obs(timeout=0.15) is True


def test_wait_for_obs_does_not_require_navigation_origin_for_non_robosuite():
    from emet.controller.generic_zmq_client import GenericZmqClient

    client = GenericZmqClient.__new__(GenericZmqClient)
    client._obs_lock = Lock()
    client._obs = {"joint": [0.0] * 4}
    client._servo = None
    client._emet_session_cache = {EMET_ZMQ_SESSION_KEY: {"runtime_kind": "molmospaces"}}
    assert client.wait_for_obs(timeout=0.15, require_navigation_origin=False) is True


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
@pytest.mark.timeout(180)
def test_wizard_records_spawn_hint_for_innate_mars():
    pytest.importorskip("mujoco")
    import mujoco

    from emet.simulation.stretch_mujoco.robocasa_gen import model_generation_wizard

    model, _xml, objects_info = model_generation_wizard(
        task="PickPlaceCounterToCabinet",
        layout=1,
        style=1,
        robot="innate_mars",
    )
    hint = objects_info.get("_emet_spawn_hint_xyt")
    assert hint is not None and len(hint) == 3
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    pos = data.body(bid).xpos[:2]
    np.testing.assert_allclose(pos, hint[:2], atol=0.02)


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
@pytest.mark.timeout(180)
def test_find_planar_base_xyt_with_hint_stays_at_robosuite_pose():
    pytest.importorskip("mujoco")
    import mujoco

    from emet.simulation import scene_base_spawn

    model, hint, _centroid, find_kw = _load_innate_mars_robocasa_kitchen()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    placed = scene_base_spawn.find_planar_base_xyt(
        data=data,
        spawn_hint_xyt=hint,
        **find_kw,
    )
    assert placed is not None
    d_hint = float(np.hypot(placed[0] - hint[0], placed[1] - hint[1]))
    assert d_hint < 0.05, f"expected spawn near robosuite hint, moved {d_hint:.3f}m"


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
@pytest.mark.timeout(180)
def test_find_planar_without_hint_prefers_walkable_clip_centroid():
    """Regression: missing hint used to snap innate_mars to clip center (~2.7, -1.5), not Stretch pose."""
    pytest.importorskip("mujoco")
    import mujoco

    from emet.simulation import scene_base_spawn

    model, hint, clip_centroid, find_kw = _load_innate_mars_robocasa_kitchen()
    hint_to_centroid = float(np.linalg.norm(hint[:2] - clip_centroid))
    if hint_to_centroid < 0.8:
        pytest.skip(f"layout places hint near clip centroid ({hint_to_centroid:.2f}m); pick another seed")

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    placed = scene_base_spawn.find_planar_base_xyt(data=data, **find_kw)
    assert placed is not None
    d_hint = float(np.hypot(placed[0] - hint[0], placed[1] - hint[1]))
    d_cent = float(np.hypot(placed[0] - clip_centroid[0], placed[1] - clip_centroid[1]))
    assert d_cent < d_hint, (
        f"without hint expected clip-centroid bias (d_cent={d_cent:.3f}m < d_hint={d_hint:.3f}m); "
        "production path must pass spawn_hint_xyt from Robosuite init_robot_base_pos"
    )
    assert d_hint > 0.5, f"without hint should move away from robosuite pose when centroid differs ({d_hint:.3f}m)"


def _innate_mars_pole_server_for_mock():
    """Lightweight RobosuiteZmqServer with planar joints (no full Robocasa kitchen)."""
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
    return RobosuiteZmqServer(
        robot_spec=spec,
        scene_model=model,
        send_port=0,
        recv_port=0,
        send_state_port=0,
        send_servo_port=0,
        use_remote_computer=False,
    )


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
@pytest.mark.timeout(120)
@patch("emet.simulation.robosuite_server.scene_base_spawn.find_planar_base_xyt")
def test_robosuite_planar_autoplace_forwards_spawn_hint_kwarg(mock_find):
    mock_find.return_value = (1.0, 2.0, 0.0)
    server = _innate_mars_pole_server_for_mock()
    server._environment_descriptor = {
        "kind": "robocasa",
        "task": "PickPlaceCounterToCabinet",
        "spawn_hint_xyt": [3.25, -0.83, 1.571],
    }
    server._load_model()
    mock_find.reset_mock()
    server._robocasa_planar_autoplace_after_load()
    mock_find.assert_called_once()
    passed = mock_find.call_args.kwargs["spawn_hint_xyt"]
    np.testing.assert_allclose(passed[:3], [3.25, -0.83, 1.571], atol=1e-6)


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
@pytest.mark.timeout(120)
@patch("emet.simulation.robosuite_server.scene_base_spawn.find_planar_base_xyt")
def test_robosuite_spawn_hint_xy_fills_yaw_from_current_base(mock_find):
    mock_find.return_value = (0.0, 0.0, 0.0)
    server = _innate_mars_pole_server_for_mock()
    server._environment_descriptor = {"kind": "robocasa", "spawn_hint_xyt": [1.5, -2.0]}
    server._load_model()
    expected_yaw = float(server.get_base_xyt()[2])
    mock_find.reset_mock()
    server._robocasa_planar_autoplace_after_load()
    passed = mock_find.call_args.kwargs["spawn_hint_xyt"]
    assert abs(float(passed[2]) - expected_yaw) < 1e-6
    np.testing.assert_allclose(passed[:2], [1.5, -2.0], atol=1e-6)
