# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from emet.controller.controller_dynamem import DynamemController


class _PlannerFail:
    def plan(self, start, goal):
        return SimpleNamespace(success=False, reason="blocked", trajectory=[])

    def clean_path_for_xy(self, waypoints):
        return waypoints


@pytest.fixture
def nav_agent(monkeypatch):
    robot = MagicMock()
    robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    robot.get_emet_session.return_value = None
    params = {"eqa": {"habitat_perfect_nav": False}}
    agent = DynamemController.__new__(DynamemController)
    agent.robot = robot
    agent.parameters = params
    agent.rerun_visualizer = MagicMock()
    agent.pos_err_threshold = 0.1
    agent.rot_err_threshold = 0.1
    agent.space = MagicMock()
    agent.space.sample_navigation.return_value = np.array([1.0, 1.0, 0.0])
    agent.planner = _PlannerFail()
    agent._last_nav_attempt = None
    agent._episode_diagnostics_recorder = None
    agent._cached_navigation_origin_xyt = None
    monkeypatch.setattr(
        "emet.controller.dynamem.eqa.habitat_perfect_nav_enabled",
        lambda _p: False,
    )
    monkeypatch.setattr(
        "emet.controller.dynamem.eqa.is_habitat_robot_client",
        lambda _r: False,
    )
    return agent


def test_navigate_to_target_pose_returns_false_on_planner_failure(nav_agent):
    finished = nav_agent.navigate_to_target_pose(
        np.array([2.0, 2.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
    )
    assert not bool(finished)
    assert nav_agent._last_nav_attempt is not None
    assert nav_agent._last_nav_attempt.finished is False
    assert nav_agent._last_nav_attempt.note == "blocked"
    nav_agent.robot.execute_trajectory.assert_not_called()


def test_navigate_to_target_pose_uses_navmesh_when_enabled(monkeypatch):
    robot = MagicMock()
    robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    params = {"eqa": {"habitat_perfect_nav": True}}
    agent = DynamemController.__new__(DynamemController)
    agent.robot = robot
    agent.parameters = params
    agent.rerun_visualizer = MagicMock()
    agent._last_nav_attempt = None
    agent._episode_diagnostics_recorder = None
    nav_res = SimpleNamespace(
        success=True,
        finished=True,
        dist_m=0.5,
        method="habitat_navmesh",
        note="ok",
        target_obs_id=None,
    )
    monkeypatch.setattr(
        "emet.controller.dynamem.eqa.habitat_perfect_nav_enabled",
        lambda _p: True,
    )
    monkeypatch.setattr(
        "emet.controller.dynamem.eqa.is_habitat_robot_client",
        lambda _r: True,
    )
    monkeypatch.setattr(
        "emet.controller.dynamem.eqa.habitat_navmesh_navigate",
        lambda *a, **k: nav_res,
    )
    finished = agent.navigate_to_target_pose(np.array([1.0, 2.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    assert bool(finished)
    assert agent._last_nav_attempt.method == "habitat_navmesh"


class _LongChunkPlanner:
    """First plan is >8 A* cells; later plans are short so the hop can finish."""

    def __init__(self):
        self.n_plan = 0
        self.plan_starts: list[np.ndarray] = []
        self._clearance_m = 0.0

    def plan(self, start, goal):
        self.n_plan += 1
        start_a = np.asarray(start, dtype=np.float64).reshape(-1)
        self.plan_starts.append(start_a[:2].copy())
        goal_a = np.asarray(goal, dtype=np.float64).reshape(-1)
        n = 20 if self.n_plan == 1 else 4
        xs = np.linspace(float(start_a[0]), float(goal_a[0]), n)
        ys = np.linspace(float(start_a[1]), float(goal_a[1]), n)
        traj = [SimpleNamespace(state=[float(xs[i]), float(ys[i]), 0.0]) for i in range(n)]
        return SimpleNamespace(success=True, trajectory=traj, reason="")

    def clean_path_for_xy(self, waypoints, start_yaw=None):
        return [list(np.asarray(w, dtype=np.float64).reshape(-1)[:3]) for w in waypoints]

    def is_explored_xy(self, xy):
        return True

    def clearance_at_xy(self, xy):
        return 1.0


def test_navigate_to_target_pose_hops_until_chunk_arrives(nav_agent, monkeypatch):
    planner = _LongChunkPlanner()
    nav_agent.planner = planner
    nav_agent._min_clearance_m = 0.0
    nav_agent.update = MagicMock()
    pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    nav_agent.robot.get_base_pose.side_effect = lambda: pose.copy()

    def _exec(traj, **kwargs):
        nonlocal pose
        finite = [
            np.asarray(p, dtype=np.float64).reshape(-1)
            for p in traj
            if np.isfinite(np.asarray(p).reshape(-1)[:2]).all()
        ]
        pose = finite[-1][:3].copy()
        return True

    nav_agent.robot.execute_trajectory.side_effect = _exec
    monkeypatch.setattr(
        "emet.controller.nav_confirm.confirm_navigation_plan",
        lambda *a, **k: True,
    )
    out = nav_agent.navigate_to_target_pose(
        np.array([2.0, 2.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        target_theta=0.3,
    )
    assert out.ok
    assert planner.n_plan >= 2
    assert nav_agent.robot.execute_trajectory.call_count >= 2
    assert nav_agent._last_nav_attempt.finished is True
    nav_agent.update.assert_called()


def test_navigate_to_target_pose_hop_uses_world_frame_start(nav_agent, monkeypatch):
    """Robocasa spawn is not world origin; hop 2 must replan from composed world XY."""
    from emet.controller.dynamem.constants import DYNAMEM_NAV_CHUNK_WPS
    from emet.utils.geometry import xyt_global_to_base

    origin = np.array([2.9, -1.7, 0.0], dtype=np.float64)
    world_start = origin.copy()
    world_goal = origin + np.array([4.0, 0.0, 0.0], dtype=np.float64)
    planner = _LongChunkPlanner()
    nav_agent.planner = planner
    nav_agent._min_clearance_m = 0.0
    nav_agent.update = MagicMock()
    nav_agent.space.sample_navigation.return_value = world_goal.copy()
    nav_agent.robot.get_emet_session.return_value = {"navigation_origin_xyt": origin.tolist()}
    nav_agent._cached_navigation_origin_xyt = origin.copy()
    local = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    nav_agent.robot.get_base_pose.side_effect = lambda: local.copy()

    def _exec(traj, **kwargs):
        nonlocal local
        finite = [
            np.asarray(p, dtype=np.float64).reshape(-1)
            for p in traj
            if np.isfinite(np.asarray(p).reshape(-1)[:2]).all()
        ]
        world_end = finite[-1][:3].copy()
        local = np.asarray(xyt_global_to_base(world_end, origin), dtype=np.float64).reshape(-1)[:3]
        return True

    nav_agent.robot.execute_trajectory.side_effect = _exec
    monkeypatch.setattr(
        "emet.controller.nav_confirm.confirm_navigation_plan",
        lambda *a, **k: True,
    )
    out = nav_agent.navigate_to_target_pose(world_goal, world_start, target_theta=0.3)
    assert out.ok
    assert planner.n_plan >= 2
    np.testing.assert_allclose(planner.plan_starts[0], world_start[:2], atol=1e-5)
    xs = np.linspace(float(world_start[0]), float(world_goal[0]), 20)
    ys = np.linspace(float(world_start[1]), float(world_goal[1]), 20)
    hop0_end = np.array([xs[DYNAMEM_NAV_CHUNK_WPS - 1], ys[DYNAMEM_NAV_CHUNK_WPS - 1]])
    np.testing.assert_allclose(planner.plan_starts[1], hop0_end, atol=0.15)
    # Episode-local GPS after hop 0 is near the chunk end minus origin, not the world cell.
    local_after_hop0 = hop0_end - origin[:2]
    assert np.linalg.norm(planner.plan_starts[1] - local_after_hop0) > 1.0


def test_process_text_empty_continues_saved_explore_traj(nav_agent, monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("empty-text explore must not pick a new frontier while leftover exists")

    monkeypatch.setattr("emet.controller.dynamem.navigation.pick_uncovered_explore_target", _boom)
    nav_agent.space.traj = [
        [1.0, 1.0, 0.0],
        [np.nan, np.nan, np.nan],
        [3.0, 4.0, 1.5],
    ]
    nav_agent.encoder = MagicMock()
    nav_agent.voxel_map = MagicMock()
    planner = _LongChunkPlanner()
    planner.n_plan = 1  # next plan() uses the short branch
    nav_agent.planner = planner
    nav_agent._min_clearance_m = 0.0
    nav_agent.robot.say = MagicMock()
    nav_agent.space.sample_navigation.return_value = np.array([3.0, 4.0, 0.0])
    nav_agent.obs_count = 0
    nav_agent.rerun_visualizer = SimpleNamespace(
        enabled=False,
        clear_nav_plan=MagicMock(),
        clear_identity=MagicMock(),
        log_nav_plan=None,
        log_arrow3D=MagicMock(),
    )
    nav_agent._rerun_refresh_monologue_panel = lambda: None  # type: ignore[method-assign]
    traj = nav_agent.process_text("", np.array([0.0, 0.0, 0.0]))
    assert len(traj) >= 2
    assert np.isnan(np.asarray(traj[-2], dtype=np.float64)).all()
    goal = np.asarray(traj[-1], dtype=np.float64).reshape(-1)
    assert abs(float(goal[0]) - 3.0) < 1e-6
    assert abs(float(goal[1]) - 4.0) < 1e-6
