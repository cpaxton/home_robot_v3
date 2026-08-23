# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CHAT find/explore/diagnostics tool returns surface nav plan outcomes."""

from __future__ import annotations

import numpy as np

from emet.agent.tools import (
    format_last_nav_plan_summary,
    format_nav_outcome_head,
    get_tools,
)


def test_format_nav_outcome_heads():
    assert "cancelled" in format_nav_outcome_head("user_cancelled", ok=False, verb="Find").lower()
    assert "aborted" in format_nav_outcome_head("aborted_waypoint_timeout", ok=False, verb="Explore").lower()
    assert "rejected_low_clearance" in format_nav_outcome_head("rejected_low_clearance", ok=False, verb="Find")
    assert format_nav_outcome_head(None, ok=True, verb="Find") == "Find finished."


def test_format_last_nav_plan_summary_fields():
    class A:
        _last_nav_plan = {
            "localize_source": "graph",
            "n_planned": 8,
            "path_m": 3.2,
            "min_clearance_m": 0.41,
            "outcome": "user_cancelled",
            "confirmed": False,
        }

    s = format_last_nav_plan_summary(A())
    assert "localize=graph" in s
    assert "planned_wps=8" in s
    assert "path≈3.20m" in s
    assert "min_clearance=0.41m" in s
    assert "outcome=user_cancelled" in s


def test_find_objects_surfaces_abort_outcome():
    class FakeAgent:
        _last_nav_plan = {
            "localize_source": "voxel",
            "n_planned": 5,
            "path_m": 1.2,
            "min_clearance_m": 0.18,
            "outcome": "aborted_waypoint_timeout",
        }
        planner = None

    class FakeExec:
        agent = FakeAgent()

        def __call__(self, cmds):
            assert cmds == [("find", "aerosol")]
            return False

    by_name = {t.name: t for t in get_tools({"executor": FakeExec(), "robot": None})}
    out = by_name["find_objects"].func(text="aerosol")
    assert "aborted" in out.lower()
    assert "Last plan:" in out
    assert "localize=voxel" in out
    assert "do not immediately re-call" in out.lower()
    assert by_name["find_objects"].returns_info is True


def test_explore_surfaces_rejected_clearance():
    class FakeVM:
        grid_origin = np.array([2.0, 2.0, 0.0])
        grid_resolution = 0.1

        def get_2d_map(self):
            o = np.zeros((5, 5), dtype=bool)
            e = np.zeros((5, 5), dtype=bool)
            e[1:4, 1:4] = True
            return o, e

    class FakeAgent:
        _last_nav_plan = {
            "localize_source": "frontier",
            "n_planned": 3,
            "min_clearance_m": 0.05,
            "outcome": "rejected_low_clearance",
        }

        def get_voxel_map(self):
            return FakeVM()

    class FakeExec:
        agent = FakeAgent()

        def __call__(self, cmds):
            return False

    by_name = {t.name: t for t in get_tools({"executor": FakeExec(), "robot": None})}
    out = by_name["explore"].func()
    assert "rejected_low_clearance" in out
    assert "Last plan:" in out


def test_list_scene_relations_falls_back_to_graph_eqa():
    class Node:
        def __init__(self, labels, xyz):
            self.labels = labels
            self.xyz = np.asarray(xyz, dtype=float)
            self.is_viewpoint = False
            self.is_frontier = False
            self.node_id = 1
            self.obs_id = 0
            self.description = None

    class FakeGM:
        def get_nodes(self):
            return [Node(["aerosol"], [1.0, 2.0, 0.5])]

        def get_edges(self):
            return []

    class FakeVM:
        def get_scene_graph(self):
            return None

    class FakeAgent:
        graph_memory = FakeGM()

        def get_voxel_map(self):
            return FakeVM()

    class FakeExec:
        agent = FakeAgent()

    by_name = {
        t.name: t
        for t in get_tools(
            {
                "executor": FakeExec(),
                "robot": None,
                "graph_memory": FakeGM(),
            }
        )
    }
    out = by_name["list_scene_relations"].func()
    assert "aerosol" in out.lower()
    assert "GraphEQA" in out
    assert "No open-vocab scene graph data yet" not in out


def test_navigation_diagnostics_includes_plan_line():
    class FakeVM:
        grid_origin = np.array([2.0, 2.0, 0.0])
        grid_resolution = 0.1

        def get_2d_map(self):
            o = np.zeros((5, 5), dtype=bool)
            e = np.ones((5, 5), dtype=bool)
            return o, e

    class FakeAgent:
        _last_nav_plan = {
            "localize_source": "graph",
            "n_planned": 4,
            "path_m": 2.0,
            "outcome": "ok",
        }
        planner = None

        def get_voxel_map(self):
            return FakeVM()

    class FakeExec:
        agent = FakeAgent()

    by_name = {t.name: t for t in get_tools({"executor": FakeExec(), "robot": None})}
    out = by_name["navigation_diagnostics"].func()
    assert "Last plan:" in out
    assert "localize=graph" in out
