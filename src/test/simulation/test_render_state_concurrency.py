# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Slow rendering must not stop measured state or physics updates."""
import threading
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from emet.simulation.robosuite_server import RobosuiteZmqServer


@pytest.mark.parametrize('method', ['_render_rgb_raw', '_render_primary_rgb_and_depth_raw'])
def test_render_releases_physics_lock_and_keeps_one_snapshot(method):
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint type="slide"/><geom size=".1"/></body></worldbody></mujoco>')
    server = RobosuiteZmqServer.__new__(RobosuiteZmqServer)
    server._mjmodel, server._mjdata = model, mujoco.MjData(model)
    mujoco.mj_forward(model, server._mjdata)
    server._mj_lock, server._render_lock = threading.RLock(), threading.RLock()
    server._camera_for_renderer = lambda _: 'camera'
    server._configure_renderer_geomgroups_for_camera = lambda *_: None
    entered, release = threading.Event(), threading.Event()
    scenes, errors = [], []

    def render():
        entered.set()
        assert release.wait(3)
        return np.zeros((2, 2, 3))

    renderer = SimpleNamespace(update_scene=lambda data, **_: scenes.append(data.qpos.copy()),
        render=render, enable_depth_rendering=lambda: None, disable_depth_rendering=lambda: None)
    server._get_or_create_primary_renderer = lambda: renderer

    def run():
        try:
            getattr(server, method)('camera')
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert entered.wait(3)
        acquired = server._mj_lock.acquire(timeout=.2)
        assert acquired, 'rendering blocks measured state and physics'
        try:
            server._mjdata.qpos[0] = .5
            mujoco.mj_forward(model, server._mjdata)
        finally:
            server._mj_lock.release()
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert not errors
    assert all(np.array_equal(q, [0.]) for q in scenes), 'RGB and depth must use the same snapshot'
