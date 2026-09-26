# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
from types import SimpleNamespace

import numpy as np

from emet.motion.navigation_sweep import validate_navigation_sweep
from emet.robots.footprint import Footprint


def test_sweep_rejects_obstacle_between_safe_endpoints():
    space = SimpleNamespace(is_valid=lambda p: not 0.45 < p[0] < 0.55, last_validity={"reason": "obstacle"})
    assert validate_navigation_sweep(space, [0, 0, 0], [[1, 0, 0]]) == (False, "obstacle")


def test_sweep_checks_rotation_and_unknown_start():
    space = SimpleNamespace(is_valid=lambda p: not 0.4 < p[2] < 0.6)
    assert not validate_navigation_sweep(space, [0, 0, 0], [[0, 0, 1]])[0]
    space.is_valid = lambda p: p[0] > 0
    assert not validate_navigation_sweep(space, [0, 0, 0], [[1, 0, 0]])[0]


def test_sweep_uses_short_yaw_arc_and_preserves_input():
    seen = []
    space = SimpleNamespace(is_valid=lambda p: seen.append(p.copy()) or True)
    start = np.array([0.0, 0.0, 3.1])
    assert validate_navigation_sweep(space, start, [[0, 0, -3.1]]) == (True, None)
    assert all(abs(p[2] - 3.1) < 0.1 for p in seen)
    assert start.tolist() == [0.0, 0.0, 3.1]


def test_sweep_rejects_nonfinite_pose():
    space = SimpleNamespace(is_valid=lambda p: True)
    assert validate_navigation_sweep(space, [0, 0, 0], [[float("nan"), 0, 0]]) == (False, "invalid_navigation_pose")


def test_conservative_footprint_contains_rotated_offset_corners():
    footprint = Footprint(length=0.33, width=0.34, length_offset=-0.1)
    for yaw in np.linspace(-np.pi, np.pi, 25):
        mask = footprint.get_conservative_rotated_mask(0.05, yaw)
        assert mask.shape[0] == mask.shape[1] and mask.shape[0] % 2 == 1
        for x in [-0.165, 0.165]:
            for y in [-0.17, 0.17]:
                bx = x - 0.1
                wx, wy = np.cos(yaw) * bx - np.sin(yaw) * y, np.sin(yaw) * bx + np.cos(yaw) * y
                i, j = np.round(np.array([wx, wy]) / 0.05).astype(int) + mask.shape[0] // 2
                assert mask[i, j]


def test_fractional_base_pose_does_not_shift_footprint_rearward():
    fp = Footprint(length=.33, width=.34, length_offset=-.1)
    pose = np.array([-1.0023702383, -.2798024416, .0185558926])
    cells = fp.grid_cells(.1, pose, [512,512])
    assert cells[:,0].min() == 499
    assert not any(tuple(cell) in {(498,507),(498,508),(498,510),(498,511)} for cell in cells)
    # A sub-cell shift changes occupied cells even when int(grid coordinate) does not.
    moved = fp.grid_cells(.1, pose + [.06,0,0], [512,512])
    assert {tuple(c) for c in moved} != {tuple(c) for c in cells}


def test_fractional_raster_is_translation_equivariant_and_conservative():
    fp = Footprint(length=.67, width=.31, length_offset=-.19, width_offset=.04)
    origin = np.array([20,30])
    for yaw in np.linspace(-np.pi, np.pi, 17):
        pose = np.array([-.023,.076,yaw])
        cells = {tuple(c) for c in fp.grid_cells(.1,pose,origin)}
        shifted = {tuple(c - [3,-2]) for c in fp.grid_cells(.1,pose+[.3,-.2,0],origin)}
        assert cells == shifted
        for x in np.linspace(-fp.length/2,fp.length/2,11):
            for y in np.linspace(-fp.width/2,fp.width/2,11):
                bx,by = x+fp.length_offset,y+fp.width_offset
                point = pose[:2] + [np.cos(yaw)*bx-np.sin(yaw)*by,np.sin(yaw)*bx+np.cos(yaw)*by]
                index = tuple(np.floor(point/.1+origin+.5).astype(int))
                assert index in cells
