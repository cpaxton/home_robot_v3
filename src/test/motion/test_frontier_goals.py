# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for top-K explore frontier collection (no MuJoCo / GPU)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from emet.motion.frontier_goals import collect_explore_frontier_candidates


class _Node:
    def __init__(self, x: float, y: float, *, frontier: bool = True):
        self.xyz = np.array([x, y, 1.0], dtype=np.float64)
        self.is_frontier = frontier
        self.cluster_size = 1
        self.region_id = None


class _Graph:
    def __init__(self, nodes: list[_Node]):
        self._nodes = nodes

    def get_nodes(self):
        return list(self._nodes)


class _FakeRobot:
    def get_base_pose(self):
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)


def test_collect_explore_frontier_candidates_seeds_and_graph_dedup():
    graph = _Graph(
        [
            _Node(2.0, 0.0),
            _Node(2.05, 0.02),  # near-dup of first
            _Node(5.0, 1.0),
            _Node(0.0, 0.0, frontier=False),
        ]
    )
    agent = SimpleNamespace(
        robot=_FakeRobot(),
        graph_memory=graph,
        planner=object(),
        space=None,
        _habitat_blocked_goals=set(),
        _habitat_recent_goals=[],
        _planning_base_xyt=lambda pose: np.asarray(pose, dtype=np.float64).reshape(-1)[:3],
        parameters={},
    )
    seed = np.array([1.0, 0.5, 1.0], dtype=np.float64)
    out = collect_explore_frontier_candidates(agent, k=8, seeds=[seed], dedup_m=0.35)
    assert len(out) >= 2
    assert any(abs(float(p[0]) - 1.0) < 1e-6 and abs(float(p[1]) - 0.5) < 1e-6 for p in out)
    # Near-dup graph nodes collapse to one cell under dedup_m.
    near = [p for p in out if abs(float(p[0]) - 2.0) < 0.2]
    assert len(near) == 1
    assert any(abs(float(p[0]) - 5.0) < 1e-6 for p in out)


def test_collect_skips_habitat_robot_client():
    class HabitatRobotClient:
        def get_base_pose(self):
            return np.zeros(3)

    agent = SimpleNamespace(
        robot=HabitatRobotClient(),
        graph_memory=_Graph([_Node(3.0, 0.0)]),
        _habitat_blocked_goals=set(),
        _habitat_recent_goals=[],
    )
    assert collect_explore_frontier_candidates(agent, k=4, seeds=[np.array([1.0, 0.0, 1.0])]) == []


def test_collect_respects_blocked_and_k():
    graph = _Graph([_Node(float(i), 0.0) for i in range(1, 12)])
    agent = SimpleNamespace(
        robot=_FakeRobot(),
        graph_memory=graph,
        planner=object(),
        space=None,
        _habitat_blocked_goals={(2.0, 0.0)},
        _habitat_recent_goals=[],
        _planning_base_xyt=lambda pose: np.asarray(pose, dtype=np.float64).reshape(-1)[:3],
        parameters={},
    )
    out = collect_explore_frontier_candidates(agent, k=3, dedup_m=0.5)
    assert len(out) == 3
    keys = {(round(float(p[0]), 2), round(float(p[1]), 2)) for p in out}
    assert (2.0, 0.0) not in keys
