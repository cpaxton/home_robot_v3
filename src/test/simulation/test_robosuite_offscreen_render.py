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
