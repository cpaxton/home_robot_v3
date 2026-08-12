# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Sourccey navigation: planar-base robot must reach a world xyt goal through the ZMQ server.

In-process test: builds a Robocasa kitchen with the vendored Sourccey MJCF, instantiates
``RobosuiteZmqServer``, drives ``handle_action({"xyt": ...})`` and verifies the base
converges to the goal (nav tolerance like other planar robots).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

_run_sim = os.environ.get("RUN_SIM_TESTS", "1").strip().lower()
RUN_SIM_TESTS = _run_sim not in ("0", "false", "no", "off")


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
@pytest.mark.timeout(240)
def test_robocasa_sourccey_navigates_to_world_goal():
    pytest.importorskip("mujoco")
    from emet.robots.sourccey import SourcceyBackend
    from emet.simulation.robosuite_server import RobosuiteZmqServer
    from emet.simulation.stretch_mujoco.robocasa_gen import model_generation_wizard

    model, _xml, objects_info = model_generation_wizard(
        task="PickPlaceCounterToCabinet", layout=1, style=1, robot="sourccey", seed=0
    )
    spec = SourcceyBackend().get_spec()
    assert spec.planar_base_joint_names is not None
    spawn_hint = np.asarray(objects_info["_emet_spawn_hint_xyt"], dtype=np.float64).reshape(-1)[:3]
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
    server._running = True

    # goal ~1 m ahead of spawn in the kitchen
    start = np.asarray(server.get_base_xyt(), dtype=np.float64)
    goal = start + np.array([1.0, 0.5, 0.1])
    with server._mj_lock:
        server.handle_action({"xyt": goal.tolist(), "nav_world": True})

    ok = False
    for _ in range(int(20.0 / 0.05)):
        with server._mj_lock:
            server._step_base_navigation_drive()
            for _ in range(server._mj_substeps_per_tick):
                server._mj_step_once()
        cur = np.asarray(server.get_base_xyt(), dtype=np.float64)
        # server deadbands XY velocity inside 2x nav_tol_xy then rotates to final yaw,
        # so accept the server's effective arrival region (2x XY tol + yaw tol).
        if np.hypot(goal[0] - cur[0], goal[1] - cur[1]) < 2 * server._nav_tol_xy and abs(goal[2] - cur[2]) < 0.2:
            ok = True
            break
    cur = np.asarray(server.get_base_xyt(), dtype=np.float64)
    assert ok, f"sourccey did not reach goal {goal.round(2)}; base={cur.round(2)}"
    server._running = False
    server._close_renderers()


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
@pytest.mark.timeout(180)
def test_robocasa_sourccey_arms_move_and_mirror():
    """Arm joints reach commanded targets and keep left/right mirror symmetry."""
    import mujoco

    from emet.robots.sourccey import SourcceyBackend
    from emet.simulation.robosuite_server import RobosuiteZmqServer
    from emet.simulation.stretch_mujoco.robocasa_gen import model_generation_wizard

    model, _xml, _ = model_generation_wizard(
        task="PickPlaceCounterToCabinet", layout=1, style=1, robot="sourccey", seed=0
    )
    spec = SourcceyBackend().get_spec()
    server = RobosuiteZmqServer(
        robot_spec=spec,
        scene_model=model,
        send_port=0,
        recv_port=0,
        send_state_port=0,
        send_servo_port=0,
        environment={"kind": "robocasa"},
    )
    server._load_model()
    server._stabilize_physics_state_after_load()
    server._initial_xyt = server.get_base_xyt()
    m = server._mjmodel
    d = server._mjdata
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    # mirrored joint targets: left uses +v, right uses -v (sagittal-mirrored chain)
    targets = {
        "left_shoulder_pan": 0.4,
        "left_shoulder_lift": -0.5,
        "left_elbow_flex": 1.4,
        "right_shoulder_pan": -0.4,
        "right_shoulder_lift": 0.5,
        "right_elbow_flex": -1.4,
        "left_gripper": 1.0,
        "right_gripper": -1.0,
    }
    idx = {jn: i for i, jn in enumerate(spec.joint_names)}
    jt = np.zeros(spec.dof)
    for jn, v in targets.items():
        jt[idx[jn]] = v
    with server._mj_lock:
        server.handle_action({"joint": jt.tolist()})
        for _ in range(400):
            server._mj_step_once()

    # each joint moved toward its target
    for jn, v in targets.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        q = float(d.qpos[m.jnt_qposadr[jid]])
        assert abs(q - v) < 0.1, f"{jn} qpos={q:.2f} target={v}"

    # left/right mirror in base frame (undo base yaw)
    byaw = server.get_base_xyt()[2]
    R = np.array([[np.cos(-byaw), -np.sin(-byaw)], [np.sin(-byaw), np.cos(-byaw)]])
    bx = d.body("base_root").xpos[:2]

    def local(p):
        return R @ (p[:2] - bx)

    l = local(d.body("left_Gripper-Base-Back-v1").xpos)
    r = local(d.body("right_Gripper-Base-Back-v1").xpos)
    assert abs(l[0] + r[0]) < 0.03 and abs(l[1] - r[1]) < 0.03, f"arms not mirrored: {l} vs {r}"
    assert np.isfinite(d.qacc).all()
    server._running = False
    server._close_renderers()
