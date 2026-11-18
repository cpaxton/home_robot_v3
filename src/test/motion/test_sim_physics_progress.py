# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Physical-mode simulation must advance, without legacy base-pose pinning."""

import threading
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from emet.simulation.physical_execution import audit_physical_action
from emet.simulation.robosuite_server import RobosuiteZmqServer


def make_server():
    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <option gravity="0 0 0" timestep=".002"/>
      <worldbody><body name="base"><freejoint/><geom size=".1"/>
        <body name="head"><joint name="tilt" axis="0 1 0" damping="2"/>
          <geom size=".02"/><camera name="head_camera" pos=".1 0 0"/>
        </body></body></worldbody>
      <actuator><position joint="tilt" kp="10"/></actuator>
      <equality><weld name="emet_stationary_base" body1="world" body2="base"/></equality>
    </mujoco>""")
    server = object.__new__(RobosuiteZmqServer)
    server._mjmodel = model
    server._mjdata = mujoco.MjData(model)
    server._mj_lock = threading.RLock()
    server._running = True
    server._physics_error = None
    server._mujoco_ctrl_debug_enabled = lambda: False
    return server


def test_physical_tick_changes_measured_joint_and_camera_without_pinning(monkeypatch):
    monkeypatch.setenv("EMET_PHYSICAL_EXECUTION", "1")
    server = make_server()
    server._hold_stationary_base_freejoint_if_idle = lambda: pytest.fail("legacy freejoint hold")
    server._hold_stationary_planar_base_if_idle = lambda: pytest.fail("legacy planar hold")
    server._reactive_controller = None
    server._nav_yaw_slew = None
    server._apply_joint_ctrl_hold_to_actuators = lambda **kw: None
    server._step_base_navigation_drive = lambda: None
    server._mj_step_once = lambda: mujoco.mj_step(server._mjmodel, server._mjdata)
    model, data = server._mjmodel, server._mjdata
    mujoco.mj_forward(model, data)
    initial_rotation = data.cam_xmat.copy()
    initial_qpos = data.qpos.copy()
    data.ctrl[0] = -0.4
    for _ in range(500):
        server._step_controlled_physics_once()
    mujoco.mj_forward(model, data)
    assert data.time > 0.99
    assert not data.eq_active[0]
    assert abs(data.qpos[-1] - initial_qpos[-1]) > 0.1
    assert not np.allclose(data.cam_xmat, initial_rotation)


def test_legacy_mode_retains_both_idle_holds(monkeypatch):
    monkeypatch.delenv("EMET_PHYSICAL_EXECUTION", raising=False)
    calls = []
    server = SimpleNamespace(
        _hold_stationary_base_freejoint_if_idle=lambda: calls.append("freejoint"),
        _hold_stationary_planar_base_if_idle=lambda: calls.append("planar"),
    )
    RobosuiteZmqServer._hold_stationary_base_if_idle(server)
    assert calls == ["freejoint", "planar"]


def test_direct_pose_hold_remains_forbidden(monkeypatch, tmp_path):
    monkeypatch.setenv("EMET_PHYSICAL_EXECUTION", "1")
    monkeypatch.setenv("EMET_PHYSICAL_AUDIT", str(tmp_path / "audit.jsonl"))
    with pytest.raises(RuntimeError, match="forbidden_physical_actuation:stationary_base_pose_hold"):
        audit_physical_action({"stationary_base_pose_hold": True}, source="test")


def test_failed_physics_stops_server_publication():
    server = make_server()
    failure = RuntimeError("physics integration failed")

    def fail():
        raise failure

    server._step_physics_tick = fail
    with pytest.raises(RuntimeError, match="physics integration failed"):
        server._sim_loop()
    assert server._physics_error is failure
    assert not server.is_running()


def test_physics_step_limit_stops_server_publication():
    server = make_server()
    server._step_physics_tick = lambda: None
    server._max_sim_steps = 1
    server._physics_steps_executed = 1
    server._sim_loop()
    assert server._physics_error is None
    assert not server.is_running()
