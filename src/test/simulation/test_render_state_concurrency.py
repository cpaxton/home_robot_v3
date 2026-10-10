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


def test_measured_state_can_publish_between_physics_substeps(monkeypatch):
    """A slow batch cannot monopolize the lock through all its substeps."""
    from emet.simulation import robosuite_server as module

    server = RobosuiteZmqServer.__new__(RobosuiteZmqServer)
    server._mj_lock = threading.RLock()
    server._running = True
    server._mj_substeps_per_tick = 3
    server._max_sim_steps = 3
    server._physics_steps_executed = 0
    server._mujoco_ctrl_debug_enabled = lambda: False
    server._step_planar_yaw_slew = lambda: None
    server._maybe_report_fall_over = lambda: None
    received = threading.Event()
    samples = []

    def read_state():
        with server._mj_lock:
            samples.append(server._physics_steps_executed)
            received.set()

    reader = threading.Thread(target=read_state, daemon=True)

    def step():
        if server._physics_steps_executed == 0:
            reader.start()
        else:
            assert received.is_set(), 'a full physics batch blocks measured telemetry'

    def yield_to_reader(_):
        assert received.wait(1), 'scheduler yield must happen outside the physics lock'

    server._step_controlled_physics_once = step
    monkeypatch.setattr(module.time, 'sleep', yield_to_reader)
    try:
        server._sim_loop()
    finally:
        reader.join(1)
    assert samples == [1]
    assert server._physics_steps_executed == 3
    assert not server._running
