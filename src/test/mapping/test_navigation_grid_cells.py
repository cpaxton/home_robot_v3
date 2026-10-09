# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from emet.mapping.grid.grid import GridParams
from emet.mapping.voxel.voxel_map_dynamem import SparseVoxelMapNavigationSpace


def test_recorded_exploration_start_and_goal_are_different_cells():
    grid = GridParams((1024, 1024), 0.1)
    space = SimpleNamespace(obstacle_map_mode="physical", voxel_map=SimpleNamespace(grid=grid))
    to_pt = SparseVoxelMapNavigationSpace.to_pt
    assert to_pt(space, [-1.0015764979, -0.2790600018]) == (502, 509)
    assert to_pt(space, [-1.1000000238, -0.300000012]) == (501, 509)
    assert to_pt(space, [1000, 0]) == (-1, -1)


@pytest.mark.parametrize("cell", [(0, 0), (1, 9), (512, 512), (1023, 1023)])
def test_cell_centers_round_trip(cell):
    grid = GridParams((1024, 1024), 0.1)
    xy = grid.grid_coords_to_xy(torch.tensor(cell))
    assert grid.xy_to_grid_cell(xy) == cell


@pytest.mark.parametrize("xy", [[float("nan"), 0], [float("inf"), 0], [-1000, 0], [1000, 0]])
def test_invalid_or_off_map_pose_is_not_clamped(xy):
    assert GridParams((1024, 1024), 0.1).xy_to_grid_cell(np.array(xy)) is None
