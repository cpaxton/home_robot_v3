# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the
# root directory of this source tree.

"""RobosuiteZmqServer navigation frame + walkable clamp (no full Robocasa load)."""

from __future__ import annotations

import numpy as np

from emet.simulation.robosuite_server import RobosuiteZmqServer
from emet.utils.geometry import xyt_base_to_global, xyt_global_to_base


def test_spawn_rel_compose_matches_geometry_helpers():
    init = np.array([2.668, -1.497, 0.25], dtype=np.float64)
    goal_ep = np.array([0.5, 0.2, -0.1], dtype=np.float64)
    via_server = RobosuiteZmqServer._spawn_rel_xyt_to_world(goal_ep, init)
    via_geom = xyt_base_to_global(goal_ep, init)
    assert np.allclose(via_server, via_geom, atol=1e-9)
    back = xyt_global_to_base(via_server, init)
    assert np.allclose(back, goal_ep, atol=1e-9)


def test_world_coords_mistaken_for_episode_rel_double_offset():
    """Episode server composes with init; passing world XY as episode-rel shifts by ~init."""
    init = np.array([2.668, -1.497, 0.0], dtype=np.float64)
    body = init.copy()
    wrong = RobosuiteZmqServer._spawn_rel_xyt_to_world(body, init)
    assert np.linalg.norm(wrong[:2] - body[:2]) > 1.0


def test_spawn_compose_jump_guard_treats_world_without_nav_world():
    """When compose would jump >8m, server logic uses raw xyt as world (matches handle_action guard)."""
    init = np.array([2.668, -1.497, 0.0], dtype=np.float64)
    cur = init.copy()
    raw = np.array([15.452, -28.793, -0.253], dtype=np.float64)
    composed = RobosuiteZmqServer._spawn_rel_xyt_to_world(raw, init)
    jump = float(np.hypot(composed[0] - cur[0], composed[1] - cur[1]))
    assert jump > 8.0
    np.testing.assert_allclose(raw[:2], [15.452, -28.793], atol=1e-3)


def test_nav_world_action_skips_spawn_compose():
    """``nav_world`` uses raw xyt as MuJoCo world — no compose with spawn (avoids double-offset)."""
    init = np.array([2.668, -1.497, 0.0], dtype=np.float64)
    goal_world = np.array([3.17, -1.2, 0.1], dtype=np.float64)
    via_spawn = RobosuiteZmqServer._spawn_rel_xyt_to_world(goal_world, init)
    assert np.linalg.norm(via_spawn[:2] - goal_world[:2]) > 0.5
    np.testing.assert_allclose(goal_world, goal_world, atol=1e-9)


def test_episode_rotate_compose_keeps_same_world_xy():
    """rotate_in_place style goals: episode (0,0,θ) -> world xy stays at spawn."""
    init = np.array([2.668, -1.497, 0.0], dtype=np.float64)
    for theta in (0.785, 1.571, 2.356):
        ep = np.array([0.0, 0.0, theta], dtype=np.float64)
        world = RobosuiteZmqServer._spawn_rel_xyt_to_world(ep, init)
        np.testing.assert_allclose(world[:2], init[:2], atol=1e-6)
        assert abs(world[2] - theta) < 1e-6


def test_freejoint_navigation_writes_linear_velocity_before_angular_velocity():
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option gravity="0 0 0" timestep="0.01"/>
          <worldbody>
            <body name="base_link">
              <freejoint/>
              <geom type="sphere" size="0.1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    data.qpos[3:7] = [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]
    mujoco.mj_forward(model, data)
    server = object.__new__(RobosuiteZmqServer)
    server._mjmodel = model
    server._mjdata = data
    server._nav_goal_world = np.array([1.0, 0.0, 0.5], dtype=np.float64)
    server._spec = type("_Spec", (), {"base_link_name": "base_link"})()
    server._nav_kp_xy = 1.0
    server._nav_v_max = 0.4
    server._nav_kp_theta = 1.0
    server._nav_w_max = 0.5
    server._nav_tol_xy = 0.02
    server._nav_tol_theta = 0.02
    server._nav_drive_debug_ticks = 0
    server._base_freejoint_addrs = lambda: (0, 0)
    server._planar_base_velocity_actuator_ids = lambda: None
    server._sim_nav_debug_enabled = lambda: False
    server.get_base_xyt = lambda: np.zeros(3, dtype=np.float64)

    server._step_base_navigation_drive()
    np.testing.assert_allclose(data.qvel[:3], [0.4, 0.0, 0.0], atol=1e-9)
    rotation_world_from_body = np.asarray(data.body("base_link").xmat).reshape(3, 3)
    expected_angular_body = rotation_world_from_body.T @ np.array([0.0, 0.0, 0.5])
    np.testing.assert_allclose(data.qvel[3:6], expected_angular_body, atol=1e-9)
    mujoco.mj_step(model, data)
    assert data.qpos[0] > 0.0


def test_freejoint_navigation_keeps_translating_inside_former_deadband():
    """Near-goal XY must still command velocity when yaw is already satisfied."""
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option gravity="0 0 0" timestep="0.01"/>
          <worldbody>
            <body name="base_link">
              <freejoint/>
              <geom type="sphere" size="0.1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    server = object.__new__(RobosuiteZmqServer)
    server._mjmodel = model
    server._mjdata = data
    server._nav_goal_world = np.array([0.12, 0.0, 0.0], dtype=np.float64)
    server._spec = type("_Spec", (), {"base_link_name": "base_link"})()
    server._nav_kp_xy = 1.0
    server._nav_v_max = 0.4
    server._nav_kp_theta = 1.0
    server._nav_w_max = 0.5
    server._nav_tol_xy = 0.07
    server._nav_tol_theta = 0.15
    server._nav_drive_debug_ticks = 0
    server._base_freejoint_addrs = lambda: (0, 0)
    server._planar_base_velocity_actuator_ids = lambda: None
    server._sim_nav_debug_enabled = lambda: False
    server.get_base_xyt = lambda: np.zeros(3, dtype=np.float64)

    server._step_base_navigation_drive()
    assert float(data.qvel[0]) > 0.05, f"expected near-goal translation, qvel={data.qvel[:3]}"


def test_freejoint_at_goal_refreshes_idle_snapshot():
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option gravity="0 0 0" timestep="0.01"/>
          <worldbody>
            <body name="base_link">
              <freejoint/>
              <geom type="sphere" size="0.1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    data.qpos[:3] = [1.0, -0.5, 0.0]
    mujoco.mj_forward(model, data)
    server = object.__new__(RobosuiteZmqServer)
    server._mjmodel = model
    server._mjdata = data
    server._nav_goal_world = np.array([1.0, -0.5, 0.0], dtype=np.float64)
    server._spec = type("_Spec", (), {"base_link_name": "base_link"})()
    server._nav_kp_xy = 1.0
    server._nav_v_max = 0.4
    server._nav_kp_theta = 1.0
    server._nav_w_max = 0.5
    server._nav_tol_xy = 0.07
    server._nav_tol_theta = 0.15
    server._nav_drive_debug_ticks = 0
    server._stationary_base_freejoint_qpos = np.zeros(7, dtype=np.float64)
    server._stationary_planar_base_qpos = None
    server._base_freejoint_addrs = lambda: (0, 0)
    server._planar_base_velocity_actuator_ids = lambda: None
    server._sim_nav_debug_enabled = lambda: False
    server.get_base_xyt = lambda: np.array([1.0, -0.5, 0.0], dtype=np.float64)

    server._step_base_navigation_drive()
    assert server._at_goal is True
    assert server._nav_goal_world is None
    np.testing.assert_allclose(server._stationary_base_freejoint_qpos[:3], [1.0, -0.5, 0.0], atol=1e-9)


def test_nav_world_wins_over_nav_relative():
    """If a buggy client sends both flags, server resolves as absolute world."""
    from emet.robots.innate_mars import InnateMarsBackend

    spec = InnateMarsBackend().get_spec()
    server = RobosuiteZmqServer(
        robot_spec=spec,
        scene_model=None,
        send_port=0,
        recv_port=0,
        send_state_port=0,
        send_servo_port=0,
    )
    server._initial_xyt = np.array([2.668, -1.497, 0.0], dtype=np.float64)
    init = server._initial_xyt
    raw = np.array([6.093, -4.119, -1.993], dtype=np.float64)
    action = {"xyt": raw.tolist(), "nav_relative": True, "nav_world": True, "step": 1}
    server.get_base_xyt = lambda: np.array([2.668, -1.497, 0.0], dtype=np.float64)  # type: ignore[method-assign]
    server.get_base_pose = lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64)  # type: ignore[method-assign]
    wx, wy, wt, meta = server._resolve_nav_goal_world_xyt(action, raw, init)
    assert meta["frame"] == "mujoco_world"
    np.testing.assert_allclose([wx, wy], raw[:2], atol=1e-6)


def test_clamp_world_nav_xyt_inside_robocasa_clip():
    from emet.robots.innate_mars import InnateMarsBackend

    spec = InnateMarsBackend().get_spec()
    server = RobosuiteZmqServer(
        robot_spec=spec,
        scene_model=None,
        send_port=0,
        recv_port=0,
        send_state_port=0,
        send_servo_port=0,
    )
    server._nav_world_clip_rect = (-1.0, 5.0, -3.0, 2.0)
    wx, wy, wt = server._clamp_world_nav_xyt(12.0, 8.0, 0.5)
    assert wx == 5.0 and wy == 2.0
    assert abs(wt - 0.5) < 1e-9
