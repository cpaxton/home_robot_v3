# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Observed-frame motion target for a simulation-only short carry diagnostic."""
import numpy as np


def short_carry_goal(tool_position, camera_pose, distance_m=.1):
    """Move horizontally toward the fixed observer while preserving lift height."""
    start, camera = np.asarray(tool_position, float), np.asarray(camera_pose, float)
    if (start.shape != (3,) or camera.shape != (4, 4) or not np.isfinite(start).all()
            or not np.isfinite(camera).all() or not np.isfinite(distance_m) or not 0 < distance_m <= .2):
        raise ValueError('invalid_short_carry_input')
    toward = camera[:2, 3]-start[:2]
    if np.linalg.norm(toward) < .05:
        toward = camera[:2, 1]
    if np.linalg.norm(toward) < .01:
        raise ValueError('short_carry_direction_unavailable')
    return start+np.r_[toward/np.linalg.norm(toward)*distance_m, 0.]
