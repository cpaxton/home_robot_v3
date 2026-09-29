# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace

import numpy as np

from emet.motion.viewpoint_selection import make_frontier_evaluator, remember_unhelpful_view, visible_unknown_cells


def test_sector_gain_respects_heading_and_wall_occlusion():
    obstacles = np.zeros((21, 21), dtype=bool)
    explored = np.ones_like(obstacles)
    explored[12:, :] = False
    kwargs = {
        "camera_xy": (10, 10),
        "fov": 0.5,
        "max_range": 8,
        "resolution": 1,
        "to_cell": lambda xy: tuple(np.floor(xy + 0.5).astype(int)),
    }
    assert visible_unknown_cells(obstacles, explored, heading=0, **kwargs)
    assert not visible_unknown_cells(obstacles, explored, heading=np.pi, **kwargs)
    obstacles[11, :] = True
    assert not visible_unknown_cells(obstacles, explored, heading=0, **kwargs)


def test_route_rejection_precedes_view_scoring_and_noop_is_not_a_view():
    obstacles = np.zeros((100, 100), dtype=bool)
    explored = np.zeros_like(obstacles)
    pose = np.eye(4)
    pose[:2, 3] = [1, 1]
    pose[:3, 2] = [1, 0, 0]
    obs = SimpleNamespace(
        camera_K=np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]]), camera_pose=pose, rgb=np.zeros((100, 100, 3))
    )
    agent = SimpleNamespace(
        robot=SimpleNamespace(get_observation=lambda: obs),
        voxel_map=SimpleNamespace(grid_resolution=0.1, max_depth=2),
        space=SimpleNamespace(
            get_navigation_map=lambda: (obstacles, explored),
            _footprint=SimpleNamespace(width=0.3, length=0.4, width_offset=0, length_offset=0),
        ),
        planner=SimpleNamespace(
            clean_path_for_xy=lambda path, **kw: path,
            to_pt=lambda xy: tuple(np.floor(np.asarray(xy) / 0.1 + 0.5).astype(int)),
        ),
        _filter_unsafe_nav_traj=lambda path, **kw: (path, None, 0.4),
    )
    start = np.array([1, 1, 0])
    goals = [[1, 1, 0], [1.4, 1, 0], [1.4, 1, 0], [1, 1, np.pi / 2]]
    diagnostics = []
    evaluate = make_frontier_evaluator(agent, start, goals, diagnostics)
    assert evaluate([[1, 1]], 0) is None
    assert diagnostics[-1]["reason"] == "already_satisfied_view"
    assert evaluate([[1, 1], [1.4, 1]], 1) > 0
    assert evaluate([[1, 1], [1.4, 1]], 2) is None
    assert diagnostics[-1]["reason"] == "duplicate_resolved_view"
    agent._filter_unsafe_nav_traj = lambda *a, **kw: ([], "rejected_swept_footprint:occupied_footprint", None)
    assert evaluate([[1, 1]], 3) is None
    assert diagnostics[-1]["reason"].endswith("occupied_footprint")
    agent._filter_unsafe_nav_traj = lambda path, **kw: (path, None, 0.4)
    agent._last_nav_plan = {"goal_xyt": goals[1]}
    remember_unhelpful_view(agent)
    explored[99, 99] = True  # Unrelated far-away evidence does not unlock retry.
    retry = make_frontier_evaluator(agent, start, goals, diagnostics)
    assert retry([[1, 1], [1.4, 1]], 1) is None
    assert diagnostics[-1]["reason"] == "unchanged_unhelpful_view"
    explored[14, 10] = True  # Local evidence changed: this view may be useful now.
    retry = make_frontier_evaluator(agent, start, goals, diagnostics)
    assert retry([[1, 1], [1.4, 1]], 1) > 0
