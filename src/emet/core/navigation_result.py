# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Measured planar arrival, independent of any particular robot or controller."""

import math


def measured_arrival(context, pose, *, xy_tolerance, yaw_tolerance):
    goal = context["resolved_goal"]
    pose = [float(v) for v in pose]
    if len(pose) != 3 or not all(math.isfinite(v) for v in pose):
        return "failed", {"reason": "invalid measured base pose"}
    xy_error = math.hypot(pose[0] - goal[0], pose[1] - goal[1])
    yaw_error = abs(math.atan2(math.sin(pose[2] - goal[2]), math.cos(pose[2] - goal[2])))
    result = {
        **context,
        "measured_pose": pose,
        "xy_error": xy_error,
        "yaw_error": yaw_error,
        "xy_tolerance": xy_tolerance,
        "yaw_tolerance": yaw_tolerance,
    }
    return ("succeeded" if xy_error <= xy_tolerance and yaw_error <= yaw_tolerance else "failed"), result
