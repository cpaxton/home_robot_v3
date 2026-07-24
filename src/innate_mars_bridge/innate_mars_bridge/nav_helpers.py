# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Pure helpers for Mars Nav2 goals (importable without nav_msgs / rclpy)."""

from __future__ import annotations

import numpy as np

# Pure relative yaw: |x|,|y| below this → prefer Nav2 Spin (works without map TF).
_YAW_ONLY_XY_EPS_M = 1e-3


def is_yaw_only_relative(xyt: list[float] | np.ndarray, *, eps_m: float = _YAW_ONLY_XY_EPS_M) -> bool:
    """True when a relative nav goal is in-place rotation (no XY translation)."""
    goal = np.asarray(xyt, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(goal[:2])) < eps_m and abs(float(goal[2])) > 1e-6
