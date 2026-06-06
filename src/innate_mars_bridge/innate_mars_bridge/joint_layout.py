# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Map Innate Mars ROS arm state + odom into the 10-DoF Emet ``RobotSpec`` joint vector."""

from __future__ import annotations

import numpy as np

INNATE_MARS_JOINT_NAMES = [
    "base_x",
    "base_y",
    "base_yaw",
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint6M",
]

_INNATE_MARS_DOF = len(INNATE_MARS_JOINT_NAMES)


def pack_innate_mars_joint_positions(
    arm_q: np.ndarray,
    *,
    base_xyt: np.ndarray | None = None,
) -> np.ndarray:
    """Build length-10 ``joint_positions`` for ZMQ (base slide + arm + gripper mimic)."""
    arm = np.asarray(arm_q, dtype=np.float64).ravel()
    out = np.zeros(_INNATE_MARS_DOF, dtype=np.float64)
    if base_xyt is not None:
        bp = np.asarray(base_xyt, dtype=np.float64).ravel()
        if bp.size >= 1:
            out[0] = float(bp[0])
        if bp.size >= 2:
            out[1] = float(bp[1])
        if bp.size >= 3:
            out[2] = float(bp[2])
    n = min(6, arm.size)
    if n > 0:
        out[3 : 3 + n] = arm[:n]
    if n >= 6:
        out[9] = float(arm[5])
    return out


def pack_innate_mars_joint_velocities(
    arm_dq: np.ndarray,
    *,
    base_xyt_vel: np.ndarray | None = None,
) -> np.ndarray:
    """Velocity vector aligned with :func:`pack_innate_mars_joint_positions`."""
    arm = np.asarray(arm_dq, dtype=np.float64).ravel()
    out = np.zeros(_INNATE_MARS_DOF, dtype=np.float64)
    if base_xyt_vel is not None:
        bv = np.asarray(base_xyt_vel, dtype=np.float64).ravel()
        n = min(3, bv.size)
        if n > 0:
            out[:n] = bv[:n]
    n = min(6, arm.size)
    if n > 0:
        out[3 : 3 + n] = arm[:n]
    if n >= 6:
        out[9] = float(arm[5])
    return out
