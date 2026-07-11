# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Robocasa freejoint spawn: Stretch / Galaxea collision-aware autoplace vs robosuite hint."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

_run_sim = os.environ.get("RUN_SIM_TESTS", "1").strip().lower()
RUN_SIM_TESTS = _run_sim not in ("0", "false", "no", "off")


def _load_robocasa_kitchen(robot: str):
    pytest.importorskip("mujoco")
    pytest.importorskip("robocasa")
    import mujoco

    from emet.robots import get_robot_spec
    from emet.simulation import scene_base_spawn
    from emet.simulation.stretch_mujoco.robocasa_gen import model_generation_wizard

    model, xml_str, objects_info = model_generation_wizard(
        task="PickPlaceCounterToCabinet",
        layout=1,
        style=1,
        robot=robot,
    )
    hint = np.asarray(objects_info["_emet_spawn_hint_xyt"], dtype=np.float64).reshape(-1)[:3]
    spec = get_robot_spec(robot)
    assert spec is not None
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(xml_str)
        mjcf_path = f.name

    margin = scene_base_spawn.planar_spawn_footprint_xy_margin_m(spec)
    clip_pad = spec.planar_spawn_clip_edge_pad_m
    if clip_pad is None:
        clip_pad = float(0.22 + 0.5 * float(spec.planar_spawn_xy_extra_margin_m))

    data_probe = mujoco.MjData(model)
    mujoco.mj_forward(model, data_probe)
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.base_link_name)
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
        "base_body_name": spec.base_link_name,
        "robot_spec": spec,
        "merged_mjcf_path": mjcf_path,
        "environment": {"kind": "robocasa"},
        "spawn_hint_xyt": hint,
    }
    return model, hint, clip_centroid, find_kw, spec, clip_pad, margin


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
@pytest.mark.timeout(180)
def test_find_robocasa_freejoint_with_hint_stretch():
    pytest.importorskip("mujoco")
    import mujoco

    from emet.simulation import scene_base_spawn

    model, hint, _centroid, find_kw, _spec, _clip_pad, _margin = _load_robocasa_kitchen("stretch")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    placed = scene_base_spawn.find_robocasa_freejoint_xyz(data=data, **find_kw)
    assert placed is not None
    x, y, z = placed
    assert z > 0.0
    d_hint = float(np.hypot(x - hint[0], y - hint[1]))
    assert d_hint < 0.35, f"stretch spawn moved {d_hint:.3f}m from robosuite hint"
    worst = scene_base_spawn.worst_robot_nonfloor_contact_dist(
        model, data, base_body_name=find_kw["base_body_name"], floor_geom_name="floor"
    )
    assert worst >= -0.01, f"expected non-penetrating spawn, worst contact dist={worst:.5f}"


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
@pytest.mark.timeout(180)
def test_find_robocasa_freejoint_with_hint_galaxea_r1():
    pytest.importorskip("mujoco")
    import mujoco

    from emet.simulation import scene_base_spawn

    model, hint, centroid, find_kw, _spec, _clip_pad, _margin = _load_robocasa_kitchen("galaxea_r1")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    placed = scene_base_spawn.find_robocasa_freejoint_xyz(data=data, **find_kw)
    assert placed is not None
    x, y, _z = placed
    d_hint = float(np.hypot(x - hint[0], y - hint[1]))
    d_cent = float(np.hypot(x - centroid[0], y - centroid[1]))
    assert d_hint < 0.35, f"galaxea spawn should respect hint ({d_hint:.3f}m from hint)"
    assert d_hint <= d_cent + 0.05, (
        f"galaxea spawn should prefer robosuite hint over clip centroid (d_hint={d_hint:.3f}m d_cent={d_cent:.3f}m)"
    )


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
@pytest.mark.timeout(180)
def test_apply_robocasa_freejoint_autoplace_stretch_inside_walkable_clip():
    pytest.importorskip("mujoco")
    import mujoco

    from emet.simulation import scene_base_spawn
    from emet.simulation.molmospaces_mobile_autoplace import apply_robocasa_freejoint_base_autoplace

    model, hint, _centroid, find_kw, spec, clip_pad, margin = _load_robocasa_kitchen("stretch")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = dict(find_kw["environment"])
    env["spawn_hint_xyt"] = hint.tolist()
    assert apply_robocasa_freejoint_base_autoplace(
        model,
        data,
        robot_spec=spec,
        base_body_name=spec.base_link_name,
        environment=env,
        debug=False,
    )
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.base_link_name)
    robot_bodies = scene_base_spawn._bodies_descending_from(model, bid)
    xy_clip_scene = scene_base_spawn.collision_scene_xy_clip_rect(
        model,
        data,
        robot_bodies=robot_bodies,
        floor_geom_name="floor",
        margin=0.55,
        max_geom_rbound=15.0,
        suppress_exterior_filter=False,
    )
    inset = float(margin) + 0.10
    xy_clip = scene_base_spawn._erode_xy_rect(xy_clip_scene, inset) if xy_clip_scene is not None else xy_clip_scene
    assert xy_clip is not None
    bx = float(data.body(spec.base_link_name).xpos[0])
    by = float(data.body(spec.base_link_name).xpos[1])
    x0, x1, y0, y1 = xy_clip
    pad = float(clip_pad)
    assert x0 + pad <= bx <= x1 - pad and y0 + pad <= by <= y1 - pad, (
        f"spawn ({bx:.3f}, {by:.3f}) outside eroded walkable clip"
    )


def _galaxea_pole_server_for_mock():
    pytest.importorskip("mujoco")
    import mujoco

    from emet.robots.galaxea_r1 import GalaxeaR1Backend
    from emet.simulation.circle_calibration import build_merged_model_with_pole_ring
    from emet.simulation.mujoco_server import _load_default_scene_with_robot
    from emet.simulation.robosuite_server import RobosuiteZmqServer

    base = _load_default_scene_with_robot("galaxea_r1")
    if base is None:
        pytest.skip("Merged galaxea_r1 scene not available")
    data0 = mujoco.MjData(base)
    mujoco.mj_forward(base, data0)
    bid = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if bid < 0:
        pytest.skip("base_link missing")
    cx0, cy0 = float(data0.body(bid).xpos[0]), float(data0.body(bid).xpos[1])
    model = build_merged_model_with_pole_ring(cx=cx0, cy=cy0)
    spec = GalaxeaR1Backend().get_spec()
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
@patch("emet.simulation.robosuite_server.scene_base_spawn.find_robocasa_freejoint_xyz")
@patch(
    "emet.simulation.robosuite_server.RobosuiteZmqServer._base_freejoint_addrs",
    return_value=(0, 6),
)
def test_robosuite_freejoint_autoplace_forwards_spawn_hint_kwarg(mock_freejoint, mock_find):
    mock_find.return_value = (1.0, 2.0, 0.5)
    server = _galaxea_pole_server_for_mock()
    server._environment_descriptor = {
        "kind": "robocasa",
        "task": "PickPlaceCounterToCabinet",
        "spawn_hint_xyt": [3.25, -0.83, 1.571],
    }
    server._load_model()
    mock_find.reset_mock()
    server._robocasa_freejoint_autoplace_after_load()
    mock_find.assert_called_once()
    passed = mock_find.call_args.kwargs["spawn_hint_xyt"]
    np.testing.assert_allclose(passed[:3], [3.25, -0.83, 1.571], atol=1e-6)
