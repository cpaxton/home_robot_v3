# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import numpy as np
import pytest

from emet.visualization.rerun import finite_nav_waypoints, nav_path_length_xy


def test_finite_nav_waypoints_skips_nan_finish_marker():
    traj = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.1],
        [np.nan, np.nan, np.nan],
        [1.5, 0.5, 1.2],
    ]
    pts = finite_nav_waypoints(traj, z=0.12)
    assert pts.shape == (3, 3)
    assert pts[0, 2] == pytest.approx(0.12)
    assert pts[2, 2] == pytest.approx(1.2)
    expected = np.hypot(1.0, 0.0) + np.hypot(0.5, 0.5)
    assert nav_path_length_xy(pts) == pytest.approx(expected)


def test_finite_nav_waypoints_empty():
    assert finite_nav_waypoints(None).shape == (0, 3)
    assert finite_nav_waypoints([]).shape == (0, 3)
    assert nav_path_length_xy(np.zeros((0, 3))) == 0.0


def test_nav_path_length_xy_l_shape():
    pts = np.array([[0.0, 0.0, 0.1], [2.0, 0.0, 0.1], [2.0, 3.0, 0.1]], dtype=np.float64)
    assert nav_path_length_xy(pts) == pytest.approx(5.0)
