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


def test_differential_route_checks_travel_heading_outside_endpoint_yaws():
    from emet.motion.navigation_sweep import differential_drive_waypoints

    start, goal = np.array([0., 0., 0.]), np.array([0., .5, 0.])
    route = differential_drive_waypoints(start, goal)
    np.testing.assert_allclose(route[0], [0, 0, np.pi / 2])
    np.testing.assert_allclose(route[-1], goal)
    steps = np.diff(np.array([start, *route])[:, :2], axis=0)
    assert np.linalg.norm(steps, axis=1).max() <= .2

    class Space:
        def is_valid(self, pose):
            return abs(pose[2]) < .5

    # Endpoint interpolation missed the turn needed to drive sideways.
    assert validate_navigation_sweep(Space(), start, [goal])[0]
    assert not validate_navigation_sweep(Space(), start, route)[0]


def test_differential_route_turn_in_place_has_single_goal():
    from emet.motion.navigation_sweep import differential_drive_waypoints

    assert differential_drive_waypoints([1, 2, 0], [1, 2, 1]) == [[1., 2., 1.]]


def test_reverse_drive_avoids_unnecessary_half_turn_with_short_segments():
    from emet.motion.navigation_sweep import differential_drive_waypoints

    route = np.array(differential_drive_waypoints([0,0,0], [-.6,0,0], allow_reverse=True))
    assert np.all(np.abs(route[:,2]) < 1e-8)
    assert np.linalg.norm(np.diff(np.vstack(([0,0,0], route))[:,:2],axis=0),axis=1).max() <= .2 + 1e-8
    np.testing.assert_allclose(route[-1], [-.6,0,0], atol=1e-12)


def test_drive_compression_keeps_corner_and_bounds_translation():
    from emet.motion.navigation_sweep import compress_drive_waypoints

    route = [[x,0,0] for x in np.linspace(.025,.4,16)]
    route += [[.4,0,yaw] for yaw in np.linspace(.05,np.pi/2,32)]
    route += [[.4,y,np.pi/2] for y in np.linspace(.025,.4,16)]
    compressed = np.array(compress_drive_waypoints([0,0,0], route))
    assert len(compressed) == 5
    np.testing.assert_allclose(compressed[2], [.4,0,np.pi/2])
    assert np.linalg.norm(np.diff(np.vstack(([0,0,0],compressed))[:,:2], axis=0),axis=1).max() <= .2 + 1e-8
