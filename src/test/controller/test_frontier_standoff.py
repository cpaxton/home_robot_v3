# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from emet.mapping.voxel.voxel_map_dynamem import SparseVoxelMapNavigationSpace


def test_frontier_can_reach_goal_cell_while_object_keeps_standoff():
    obstacles = torch.zeros((6, 2), dtype=torch.bool)
    space = SimpleNamespace(
        voxel_map=SimpleNamespace(get_2d_map=lambda: (obstacles, ~obstacles)),
        compute_theta=lambda x, y, px, py: float(np.arctan2(py - y, px - x)),
        is_valid=lambda pose: True,
        _line_of_sight_clear=lambda *args: True,
    )
    planner = SimpleNamespace(
        to_pt=lambda pose: (round(float(pose[0]) * 10), round(float(pose[1]) * 10)),
        to_xy=lambda ij: (ij[0] / 10, ij[1] / 10),
        get_reachable_points=lambda start: [(i, 0) for i in range(6)],
    )
    sample = SparseVoxelMapNavigationSpace.sample_target_point
    target = np.array([0.5, 0, 0])
    start = np.zeros(3)
    exploration = sample(space, start, target, planner, exploration=True)
    object_goal = sample(space, start, target, planner, exploration=False)
    np.testing.assert_allclose(exploration[:2], target[:2])
    assert np.linalg.norm(object_goal[:2] - target[:2]) > 0.35
    obstacles[5, 0] = True
    blocked = sample(space, start, target, planner, exploration=True)
    assert not np.allclose(blocked[:2], target[:2])
    excluded = sample(space, start, target, planner, exploration=True, blocked={(0.4, 0.0), (0.5, 0.0)})
    assert excluded[0] < 0.4
    space._line_of_sight_clear = lambda *args: False
    assert sample(space, start, target, planner, exploration=True) is None
    # A tabletop object can be visible above occupied 2D cells. Its approach
    # does not require a collision-free ray all the way to the object itself.
    assert sample(space, start, target, planner) is None  # Legacy behavior.
    assert sample(space, start, target, planner, require_planar_visibility=False) is not None
    space._line_of_sight_clear = lambda *args: True
    space.is_valid = lambda pose: False
    assert sample(space, start, target, planner) is None
    assert sample(space, start, target, planner, require_planar_visibility=False) is None


def test_manipulation_distance_bounds_never_fall_back_to_a_close_viewpoint():
    obstacles = torch.zeros((12, 2), dtype=torch.bool)
    space = SimpleNamespace(
        voxel_map=SimpleNamespace(get_2d_map=lambda: (obstacles, ~obstacles)),
        compute_theta=lambda *args: 0.0,
        is_valid=lambda pose: True,
        _line_of_sight_clear=lambda *args: False,
    )
    planner = SimpleNamespace(
        to_pt=lambda pose: (round(float(pose[0]) * 10), 0),
        to_xy=lambda ij: (ij[0] / 10, 0),
        get_reachable_points=lambda start: [(i, 0) for i in range(12)],
    )
    sample = SparseVoxelMapNavigationSpace.sample_target_point
    kwargs = {"distance_range": (0.7, 0.8), "require_planar_visibility": False}
    goal = sample(space, np.array([1, 0, 0]), np.zeros(3), planner, **kwargs)
    assert goal[0] == 0.8
    obstacles[8, 0] = True
    assert sample(space, np.array([1, 0, 0]), np.zeros(3), planner, **kwargs) is None
    # The unmodified find policy can still use a closer viewing location.
    assert sample(space, np.ones(3), np.zeros(3), planner, require_planar_visibility=False) is not None
    obstacles[8, 0] = False
    space.is_valid = lambda pose: False
    assert sample(space, np.ones(3), np.zeros(3), planner, **kwargs) is None


@pytest.mark.parametrize("bounds", [(0.8, 0.7), (-1, 1), (0, np.inf), (np.nan, 1), (0.7,)])
def test_invalid_manipulation_distance_bounds_fail_before_planning(bounds):
    with pytest.raises(ValueError, match="distance range"):
        SparseVoxelMapNavigationSpace.sample_target_point(None, None, None, None, distance_range=bounds)
