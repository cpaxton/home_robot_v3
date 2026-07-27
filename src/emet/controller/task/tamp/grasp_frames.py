# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Grasp frame helpers for kinematic pick (approach along -Z of the grasp frame)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def top_down_grasp_T(
    xyz: np.ndarray | Sequence[float],
    *,
    z_offset_m: float = 0.02,
) -> np.ndarray:
    """World grasp pose with approach from above.

    ``KinematicPickPlaceExecutor`` uses ``approach = -R[:, 2]`` for pregrasp standoff, so the
    grasp frame's +Z must point **into** the object (world -Z) for a top-down lift.
    """
    p = np.asarray(xyz, dtype=np.float64).reshape(3).copy()
    p[2] += float(z_offset_m)
    T = np.eye(4, dtype=np.float64)
    # Columns: X right, Y back, Z down (det = +1).
    T[:3, 0] = (1.0, 0.0, 0.0)
    T[:3, 1] = (0.0, -1.0, 0.0)
    T[:3, 2] = (0.0, 0.0, -1.0)
    T[:3, 3] = p
    return T
