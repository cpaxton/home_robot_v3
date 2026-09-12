# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace

import numpy as np
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
