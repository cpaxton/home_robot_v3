# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
import numpy as np
import pytest

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.motion.observed_object_bound import enclose_observed_object
from emet.motion.observed_workspace import ObservedWorkspace


def bound(grid, points=((2.5, 2.5, 2.5),), labels=("target",)):
    return enclose_observed_object(
        grid.partition(points, np.array(labels)),
        "target",
        observation_ids=("camera:1",),
        received_monotonic=10.0,
        scene_revision="scan:1",
        topology="connected_rigid",
    )


def test_free_space_enclosure_contains_hidden_extent_not_just_visible_points():
    cells = np.ones((5, 5, 5), np.uint8)
    cells[1:4, 1:4, 1:4] = 0
    cells[2, 2, 2] = 2
    result = bound(ObservedWorkspace(np.zeros(3), 1.0, cells))
    np.testing.assert_array_equal(result.bounds_world, [[1, 1, 1], [4, 4, 4]])
    assert len(result.vertices_world) == 8
    result.require(now=11.0)
    with pytest.raises(ObservationUnavailable):
        result.require(now=13.0)
    with pytest.raises(ValueError):
        result.bounds_world.setflags(write=True)


def test_unseen_extension_reaching_scan_boundary_is_not_a_complete_volume():
    cells = np.ones((5, 5, 5), np.uint8)
    cells[2, 2, 2] = 2
    cells[2:, 2, 2] = 0
    cells[2, 2, 2] = 2
    with pytest.raises(ObservationUnavailable, match="unbounded"):
        bound(ObservedWorkspace(np.zeros(3), 1.0, cells))
    # A new measured free barrier, not a bounding-box assumption, closes it.
    second = np.zeros_like(cells)
    second[3:, 2, 2] = 1
    fused = ObservedWorkspace.fuse_static(
        [ObservedWorkspace(np.zeros(3), 1.0, cells), ObservedWorkspace(np.zeros(3), 1.0, second)]
    )
    assert bound(fused).bounds_world[1, 0] == 3


def test_diagonal_unknown_and_background_cells_cannot_be_carved_away():
    cells = np.ones((5, 5, 5), np.uint8)
    cells[2, 2, 2] = 2
    cells[3, 3, 3] = 2
    cells[4, 4, 4] = 2
    grid = ObservedWorkspace(np.zeros(3), 1.0, cells)
    with pytest.raises(ObservationUnavailable, match="unbounded"):
        bound(grid, ((2.5, 2.5, 2.5), (3.5, 3.5, 3.5), (4.5, 4.5, 4.5)), ("target", "background", "background"))
