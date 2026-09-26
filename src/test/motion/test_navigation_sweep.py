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
