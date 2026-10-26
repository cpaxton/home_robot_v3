# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
import numpy as np
import pytest

from emet.motion.observed_workspace import ObservedWorkspace


def build(depth=None):
    return ObservedWorkspace.from_depth(np.full((21, 21), 1.) if depth is None else depth,
        [[10, 0, 10], [0, 10, 10], [0, 0, 1]], np.eye(4),
        workspace=[[-.2, -.2, .2], [.2, .2, 1.4]], resolution=.1)


def test_measured_depth_carves_front_but_never_behind_surface():
    grid = build()
    assert grid.known_free([[-.05, -.05, .4], [.05, .05, .6]])
    assert not grid.known_free([[-.05, -.05, 1.1], [.05, .05, 1.2]])
    assert np.any(grid.cells == 2) and np.any(grid.cells == 0)
    scene = grid.placement_scene()
    assert not scene.collides([[-.05, -.05, .4], [.05, .05, .6]])
    assert scene.collides([[-.05, -.05, 1.1], [.05, .05, 1.2]])


@pytest.mark.parametrize('missing', [0., np.nan, np.inf, 1e30, .01])
def test_missing_return_blocks_whole_volume(missing):
    depth = np.full((21, 21), 1.)
    depth[10, 10] = missing
    grid = build(depth)
    assert not grid.known_free([[-.05, -.05, .4], [.05, .05, .6]])


def test_cell_boundaries_and_workspace_bounds_are_conservative():
    cells = np.ones((3, 3, 3), dtype=np.uint8)
    cells[2, 1, 1] = 0
    grid = ObservedWorkspace(np.zeros(3), 1., cells)
    assert grid.known_free([[1.1]*3, [1.9]*3])
    assert not grid.known_free([[1.1]*3, [2., 1.9, 1.9]])
    assert not grid.known_free([[-.01]*3, [.5]*3])
    with pytest.raises(ValueError):
        grid.cells.setflags(write=True)


def test_snapshot_keeps_sensor_holes_and_projection_clipping_unknown():
    grid = ObservedWorkspace.from_depth(np.ones((3, 3)), [[10, 0, 1], [0, 10, 1], [0, 0, 1]],
        np.eye(4), workspace=[[-.2, -.2, .2], [.2, .2, .6]], resolution=.1)
    assert not grid.known_free([[-.2, -.2, .2], [.2, .2, .6]])


def test_foreground_pixel_prevents_background_free_space_carving():
    depth = np.full((21, 21), 1.)
    depth[10, 10] = .3
    assert not build(depth).known_free([[-.05, -.05, .4], [.05, .05, .6]])
