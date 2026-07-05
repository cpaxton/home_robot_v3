# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import numpy as np

from emet.controller.habitat_nav import (
    habitat_navmesh_navigate,
    habitat_perfect_nav_enabled,
    is_habitat_robot_client,
    navmesh_waypoints_to_xyt,
    pick_habitat_exploration_target,
)


def test_habitat_perfect_nav_enabled_reads_eqa_block():
    assert habitat_perfect_nav_enabled({"eqa": {"habitat_perfect_nav": True}})
    assert not habitat_perfect_nav_enabled({"eqa": {"habitat_perfect_nav": False}})
    assert habitat_perfect_nav_enabled({"eqa": {"habitat_navmesh_nav": True}})


def test_navmesh_waypoints_to_xyt_yaw_follows_segment():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    traj = navmesh_waypoints_to_xyt(pts)
    assert len(traj) == 3
    assert abs(float(traj[0][2]) - 0.0) < 1e-6
    assert abs(float(traj[1][2]) - np.pi / 2) < 1e-5


class _FakeSim:
    def snap_navmesh_xz(self, x, z):
        return float(x), float(z), True

    def find_path_to_xy(self, x: float, z: float):
        return np.array([[0.0, 0.0, 0.0], [float(x), 0.0, float(z)]], dtype=np.float64)


class _FakeHabitatRobot:
    def __init__(self):
        self._sim = _FakeSim()
        self._pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.traj = None

    def get_base_pose(self):
        return self._pose.copy()

    def move_base_to(self, xyt, **kwargs):
        xyt = np.asarray(xyt, dtype=np.float64).reshape(-1)
        self._pose = xyt[:3].copy()
        self.traj = [self._pose.copy()]

    def execute_trajectory(self, trajectory, **kwargs):
        for wp in trajectory:
            self.move_base_to(wp, **kwargs)


def test_is_habitat_robot_client_by_sim_api():
    assert is_habitat_robot_client(_FakeHabitatRobot())


def test_habitat_navmesh_navigate_moves_robot():
    robot = _FakeHabitatRobot()
    res = habitat_navmesh_navigate(robot, np.array([1.0, 1.0]))
    assert res.method == "habitat_navmesh"
    assert res.finished
    assert res.note.startswith("ok")
    assert robot.traj is not None


def test_resolve_habitat_nav_goal_uses_path_end():
    class _Sim:
        def snap_navmesh_xz(self, x, z):
            return float(x), float(z), True

        def find_path_to_xy(self, x, z):
            return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0]], dtype=np.float64)

    from emet.controller.habitat_nav import resolve_habitat_nav_goal

    r = resolve_habitat_nav_goal(_Sim(), 5.0, 5.0)
    assert r is not None
    assert r.mode == "path_end"
    assert r.effective_xy == (1.0, 2.0)


def test_pick_habitat_exploration_target_prefers_nearby_frontier():
    class _Node:
        def __init__(self, obs_id, x, z, failures=0):
            self.obs_id = obs_id
            self.xyz = [x, z, 0.0]
            self.is_frontier = True
            self.nav_failures = failures
            self.last_seen = obs_id

    class _GM:
        def get_nodes(self):
            return [_Node(1, 10.0, 0.0), _Node(2, 1.0, 0.0)]

    class _Sim:
        def snap_navmesh_xz(self, x, z):
            return float(x), float(z), True

        def find_path_to_xy(self, x, z):
            return np.array([[0.0, 0.0, 0.0], [float(x), 0.0, float(z)]], dtype=np.float64)

    class _Robot:
        def __init__(self):
            self._sim = _Sim()
            self._pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        def get_base_pose(self):
            return self._pose.copy()

    agent = type("A", (), {"graph_memory": _GM(), "robot": _Robot()})()
    pt = pick_habitat_exploration_target(agent)
    assert pt is not None
    assert float(pt[0]) == 1.0
    assert float(pt[1]) == 0.0


def test_pick_habitat_exploration_target_skips_recent_goal():
    class _Node:
        def __init__(self, obs_id, x, z):
            self.obs_id = obs_id
            self.xyz = [x, z, 0.0]
            self.is_frontier = True
            self.nav_failures = 0
            self.last_seen = obs_id

    class _GM:
        def get_nodes(self):
            return [_Node(1, 1.0, 0.0), _Node(2, 4.0, 0.0)]

    class _Sim:
        def snap_navmesh_xz(self, x, z):
            return float(x), float(z), True

        def find_path_to_xy(self, x, z):
            return np.array([[0.0, 0.0, 0.0], [float(x), 0.0, float(z)]], dtype=np.float64)

    class _Robot:
        def __init__(self):
            self._sim = _Sim()
            self._pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        def get_base_pose(self):
            return self._pose.copy()

    agent = type(
        "A",
        (),
        {
            "graph_memory": _GM(),
            "robot": _Robot(),
            "_habitat_recent_goals": [(1.0, 0.0)],
        },
    )()
    pt = pick_habitat_exploration_target(agent)
    assert pt is not None
    assert float(pt[0]) == 4.0


def test_apply_habitat_nav_resolution_returns_effective_xy():
    from emet.controller.habitat_nav import apply_habitat_nav_resolution

    class _Sim:
        def snap_navmesh_xz(self, x, z):
            return float(x), float(z), True

        def find_path_to_xy(self, x, z):
            return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0]], dtype=np.float64)

    robot = type("R", (), {"_sim": _Sim()})()
    pt = apply_habitat_nav_resolution(robot, np.array([5.0, 5.0]))
    assert pt is not None
    assert float(pt[0]) == 1.0
    assert float(pt[1]) == 2.0
