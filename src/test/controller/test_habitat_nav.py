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


def test_habitat_navmesh_navigate_rejects_noop_when_already_at_goal():
    robot = _FakeHabitatRobot()
    res = habitat_navmesh_navigate(robot, np.array([0.05, 0.05]))
    assert res.note.startswith("already_at_goal")
    assert not res.finished
    assert res.dist_m == 0.0


def test_habitat_navmesh_navigate_moves_to_nearby_goal():
    robot = _FakeHabitatRobot()
    res = habitat_navmesh_navigate(robot, np.array([0.35, 0.0]))
    assert res.finished
    assert res.dist_m >= 0.08
    assert res.note.startswith("ok")


def test_pick_habitat_exploration_target_accepts_nearby_frontier():
    class _Node:
        def __init__(self, obs_id, x, z):
            self.obs_id = obs_id
            self.xyz = [x, z, 0.0]
            self.is_frontier = True
            self.nav_failures = 0
            self.last_seen = obs_id

    class _GM:
        def get_nodes(self):
            # 0.5m is beyond Habitat noop radius (~0.28m); 0.1m would be rejected as already_at_goal.
            return [_Node(1, 0.5, 0.0), _Node(2, 4.0, 0.0)]

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

    agent = type("A", (), {"graph_memory": _GM(), "robot": _Robot(), "parameters": {"eqa": {}}})()
    pt = pick_habitat_exploration_target(agent)
    assert pt is not None
    assert float(pt[0]) == 0.5


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


def test_pick_uncovered_skips_candidate_that_path_end_snaps_into_blocked():
    """Coverage hints that remapping into a stuck XY must not win."""
    from emet.controller.habitat_nav import pick_uncovered_explore_target

    class _Node:
        def __init__(self, obs_id, x, z):
            self.obs_id = obs_id
            self.xyz = [x, z, 0.0]
            self.is_frontier = True
            self.nav_failures = 0
            self.last_seen = obs_id

    class _GM:
        def get_nodes(self):
            return [_Node(1, 7.9, 2.8), _Node(2, 4.0, 0.0)]

    class _Sim:
        def snap_navmesh_xz(self, x, z):
            return float(x), float(z), True

        def find_path_to_xy(self, x, z):
            # Remap frontier (7.9, 2.8) onto stuck goal (-8.6, 1.2); keep other goals raw.
            if abs(float(x) - 7.9) < 0.2 and abs(float(z) - 2.8) < 0.2:
                return np.array([[0.0, 0.0, 0.0], [-8.6, 0.0, 1.2]], dtype=np.float64)
            return np.array([[0.0, 0.0, 0.0], [float(x), 0.0, float(z)]], dtype=np.float64)

    class _Robot:
        def __init__(self):
            self._sim = _Sim()
            self._pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        def get_base_pose(self):
            return self._pose.copy()

    blocked = {(-8.6, 1.2)}
    agent = type(
        "A",
        (),
        {
            "graph_memory": _GM(),
            "robot": _Robot(),
            "_habitat_blocked_goals": blocked,
            "_habitat_recent_goals": [],
            "parameters": {"eqa": {}},
            "_planning_base_xyt": lambda self, pose: pose,
        },
    )()
    cand = np.array([7.9, 2.8, 1.0], dtype=float)
    pt = pick_uncovered_explore_target(
        agent,
        question="bed",
        candidates=[cand],
        blocked=blocked,
    )
    assert pt is not None
    assert abs(float(pt[0]) - (-8.6)) > 0.5
    assert float(pt[0]) == 4.0


def test_pick_uncovered_mujoco_skips_blocked_and_uses_sample_frontier():
    from emet.controller.habitat_nav import pick_uncovered_explore_target

    class _Node:
        def __init__(self, x, z):
            self.obs_id = 1
            self.xyz = [x, z, 0.0]
            self.is_frontier = True
            self.nav_failures = 0
            self.last_seen = 1

    class _GM:
        def get_nodes(self):
            return [_Node(1.0, 0.0)]

    class _Space:
        def __init__(self):
            self.calls = 0

        def sample_frontier(self, planner, start, text=None):
            self.calls += 1
            return np.array([3.5, 1.0], dtype=float)

    class _Robot:
        def get_base_pose(self):
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    space = _Space()
    blocked = {(1.0, 0.0)}
    agent = type(
        "A",
        (),
        {
            "graph_memory": _GM(),
            "robot": _Robot(),  # no _sim => not Habitat
            "space": space,
            "planner": object(),
            "_habitat_blocked_goals": blocked,
            "_habitat_recent_goals": [],
            "_planning_base_xyt": lambda self, pose: pose,
            "_best_frontier_point_from_graph": lambda self, q: None,
        },
    )()
    pt = pick_uncovered_explore_target(agent, question="fan", blocked=blocked)
    assert pt is not None
    assert float(pt[0]) == 3.5
    assert space.calls >= 1


def test_log_nav_attempt_records_already_at_goal_into_recent():
    from emet.controller.controller_dynamem import DynamemController
    from emet.controller.habitat_nav import NavAttemptResult

    agent = DynamemController.__new__(DynamemController)
    agent._habitat_recent_goals = []
    agent._habitat_blocked_goals = set()
    agent._episode_diagnostics_recorder = None
    nav = NavAttemptResult(
        success=False,
        finished=False,
        dist_m=0.0,
        method="habitat_navmesh",
        note="already_at_goal_0.04m",
        goal_xy=(-8.6, 1.2),
        effective_goal_xy=(-8.6, 1.2),
    )
    DynamemController._log_nav_attempt(
        agent,
        nav,
        target_obs_id=None,
        goal_xy=np.array([-8.6, 1.2], dtype=float),
    )
    assert (-8.6, 1.2) in agent._habitat_recent_goals
    assert (-8.6, 1.2) in agent._habitat_blocked_goals
