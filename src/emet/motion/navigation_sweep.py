# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Sampled footprint validation for executable base-motion segments."""

import numpy as np


def validate_navigation_sweep(space, start, waypoints, *, linear_step_m=0.025, angular_step_rad=0.05):
    """Check translation and the shortest yaw sweep, including the measured start.

    Check every sampled yaw at each sampled XY: controllers may overlap turning
    and translation. This is deliberately conservative, not a dynamics/contact
    proof. Unknown footprint cells remain invalid under the space's contract.
    """
    if linear_step_m <= 0 or angular_step_rad <= 0:
        raise ValueError("Sweep resolution must be positive")
    previous = np.asarray(start, dtype=float)
    for raw in waypoints:
        target = np.asarray(raw, dtype=float)
        if previous.shape != (3,) or target.shape != (3,) or not np.isfinite([previous, target]).all():
            return False, "invalid_navigation_pose"
        yaw_delta = np.arctan2(np.sin(target[2] - previous[2]), np.cos(target[2] - previous[2]))
        n_xy = max(1, int(np.ceil(np.linalg.norm(target[:2] - previous[:2]) / linear_step_m)))
        n_yaw = max(1, int(np.ceil(abs(yaw_delta) / angular_step_rad)))
        for fraction in np.linspace(0, 1, n_xy + 1):
            xy = previous[:2] + fraction * (target[:2] - previous[:2])
            for yaw in np.linspace(previous[2], previous[2] + yaw_delta, n_yaw + 1):
                if not space.is_valid(np.array([*xy, yaw])):
                    return False, getattr(space, "last_validity", {}).get("reason", "invalid_footprint")
        previous = target
    return True, None
