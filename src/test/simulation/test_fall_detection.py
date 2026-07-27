# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for MuJoCo base fall-over detection."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from emet.simulation.fall_detection import (
    FallOverMonitor,
    assess_base_upright,
    body_up_dot_world_z,
)


class _FakeModel:
    def __init__(self, body_id: int = 0):
        self.nbody = 2
        self._body_id = body_id


class _FakeData:
    def __init__(self, up_dot_z: float, xyz=(0.0, 0.0, 0.5), sim_time: float = 2.0):
        # xmat is row-major 3x3; body Z axis is row 2.
        r = np.eye(3, dtype=np.float64)
        # Rotate about X so body Z tilts in YZ: cos(tilt) = up_dot_z
        tilt = float(np.arccos(np.clip(up_dot_z, -1.0, 1.0)))
        c, s = np.cos(tilt), np.sin(tilt)
        r = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
        self.xmat = np.zeros((2, 9), dtype=np.float64)
        self.xmat[0] = r.reshape(9)
        self.xpos = np.zeros((2, 3), dtype=np.float64)
        self.xpos[0] = np.asarray(xyz, dtype=np.float64)
        self.time = float(sim_time)


def test_body_up_dot_world_z(monkeypatch):
    class _Mj:
        class mjtObj:
            mjOBJ_BODY = 1

        @staticmethod
        def mj_name2id(model, _obj, name):
            return 0 if name == "base_link" else -1

    monkeypatch.setitem(__import__("sys").modules, "mujoco", _Mj)

    data = _FakeData(up_dot_z=0.5)
    assert abs(body_up_dot_world_z(_FakeModel(), data, "base_link") - 0.5) < 1e-6


def test_assess_upright_vs_fallen(monkeypatch):
    class _Mj:
        class mjtObj:
            mjOBJ_BODY = 1

        @staticmethod
        def mj_name2id(model, _obj, name):
            return 0 if name == "base_link" else -1

    monkeypatch.setitem(__import__("sys").modules, "mujoco", _Mj)

    ok = assess_base_upright(_FakeModel(), _FakeData(0.95), max_tilt_deg=55.0)
    assert ok.upright is True
    fallen = assess_base_upright(_FakeModel(), _FakeData(0.2), max_tilt_deg=55.0)
    assert fallen.upright is False
    assert fallen.tilt_deg > 55.0


def test_fall_monitor_logs_once(monkeypatch):
    class _Mj:
        class mjtObj:
            mjOBJ_BODY = 1

        @staticmethod
        def mj_name2id(model, _obj, name):
            return 0

    monkeypatch.setitem(__import__("sys").modules, "mujoco", _Mj)

    errors: list[str] = []
    fake_log = SimpleNamespace(
        error=lambda msg: errors.append(str(msg)),
        alert=lambda msg: None,
    )
    mon = FallOverMonitor(
        base_body_name="base_link",
        max_tilt_deg=55.0,
        min_sim_time_s=0.0,
        repeat_interval_s=1000.0,
        log=fake_log,  # type: ignore[arg-type]
    )
    model = _FakeModel()
    mon.maybe_report(model, _FakeData(0.1, sim_time=2.0))
    mon.maybe_report(model, _FakeData(0.1, sim_time=2.1))
    assert len(errors) == 1
    assert "FALLEN OVER" in errors[0]
