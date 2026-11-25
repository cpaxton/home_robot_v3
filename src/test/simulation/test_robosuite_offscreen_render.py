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

"""RobosuiteZmqServer offscreen GL helpers (no GPU render required)."""

from __future__ import annotations

import os

import mujoco
import pytest

from emet.simulation.mujoco_server import _load_default_scene_with_robot
from emet.simulation.robosuite_server import _PRIMARY_RH, _PRIMARY_RW, RobosuiteZmqServer


def test_configure_offscreen_gl_defaults_to_egl(monkeypatch):
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    RobosuiteZmqServer._configure_offscreen_gl(use_glx=False)
    assert os.environ["MUJOCO_GL"] == "egl"


def test_configure_offscreen_gl_respects_existing(monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    RobosuiteZmqServer._configure_offscreen_gl(use_glx=False)
    assert os.environ["MUJOCO_GL"] == "osmesa"


def test_ensure_offscreen_framebuffer_grows_model_buffer():
    model = _load_default_scene_with_robot("innate_mars")
    if model is None:
        pytest.skip("default innate_mars scene not available")
    vis = model.vis.global_
    vis.offwidth = 320
    vis.offheight = 240
    server = object.__new__(RobosuiteZmqServer)
    server._mjmodel = model
    server._primary_renderer = None
    server._ensure_offscreen_framebuffer(_PRIMARY_RW, _PRIMARY_RH)
    assert int(vis.offwidth) >= _PRIMARY_RW
    assert int(vis.offheight) >= _PRIMARY_RH


def test_is_offscreen_render_gl_error_detects_framebuffer_codes():
    assert RobosuiteZmqServer._is_offscreen_render_gl_error(mujoco.FatalError("Offscreen framebuffer 0x8cdd"))
    assert not RobosuiteZmqServer._is_offscreen_render_gl_error(ValueError("other"))


def test_sensor_scene_excludes_debug_sites_but_retains_physical_geometry(monkeypatch):
    from types import SimpleNamespace

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <geom name="obstacle" type="box" pos="0 0 1" size=".1 .1 .1"/>
        <site name="debug" type="cylinder" size=".005 10" rgba="0 1 0 .3" group="1"/>
        </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    server = object.__new__(RobosuiteZmqServer)
    server._mjmodel, server._primary_renderer = model, None
    monkeypatch.setattr(mujoco, "Renderer", lambda *args: SimpleNamespace(_scene_option=mujoco.MjvOption()))
    renderer = server._get_or_create_primary_renderer()
    scene = mujoco.MjvScene(model, maxgeom=20)
    mujoco.mjv_updateScene(model, data, renderer._scene_option, None, mujoco.MjvCamera(),
                          mujoco.mjtCatBit.mjCAT_ALL, scene)
    assert not any(g.objtype == mujoco.mjtObj.mjOBJ_SITE for g in scene.geoms[:scene.ngeom])
    assert any(g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid == model.geom("obstacle").id
               for g in scene.geoms[:scene.ngeom])
    assert server._get_or_create_primary_renderer() is renderer
