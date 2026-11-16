# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Native snapshots keep their model identity without hashing meshes per tick."""

from types import SimpleNamespace

import mujoco

from emet.controller.manipulation import native_scene
from emet.simulation.stretch_mujoco.mujoco_server import MujocoServer


def model(size):
    m = mujoco.MjModel.from_xml_string(
        f'<mujoco><worldbody><body><freejoint/><geom type="sphere" size="{size}"/></body></worldbody></mujoco>'
    )
    d = mujoco.MjData(m)
    d.time = 1
    return m, d


def server(m):
    instance = object.__new__(MujocoServer)
    instance.mjmodel = m
    instance._native_signature_model = m
    instance._native_model_signature = native_scene.model_signature(m)
    instance.physics_fps_counter = SimpleNamespace(tick=lambda **kwargs: None)
    instance.pull_status = lambda: None
    instance._consume_commands = lambda: None
    return instance


def test_repeated_physics_updates_reuse_compiled_model_identity(monkeypatch):
    m, d = model(0.1)
    sim = server(m)
    expected = sim._native_model_signature

    def no_rehash(_):
        raise AssertionError("Physics tick rehashed the loaded model")

    monkeypatch.setattr(native_scene, "model_signature", no_rehash)
    for _ in range(3):
        d.qpos[0] += 0.1
        sim._ctrl_callback(m, d)
    assert sim._native_model_signature == expected


def test_replacing_model_refreshes_identity_once(monkeypatch):
    original, _ = model(0.1)
    replacement, d = model(0.2)
    sim = server(original)
    first = sim._native_model_signature
    expected = native_scene.model_signature(replacement)
    real = native_scene.model_signature
    calls = []

    def count(m):
        calls.append(m)
        return real(m)

    monkeypatch.setattr(native_scene, "model_signature", count)
    sim._ctrl_callback(replacement, d)
    sim._ctrl_callback(replacement, d)
    assert calls == [replacement]
    assert sim._native_model_signature == expected != first
