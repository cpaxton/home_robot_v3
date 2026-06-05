# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Joint vector layout helpers for Stretch + Robocasa MuJoCo (``RobosuiteZmqServer``)."""

from __future__ import annotations

import numpy as np

from emet.motion.kinematics import HelloStretchIdx
from emet.robots.stretch import STRETCH_ROBOCASA_MJCF_JOINT_NAMES

_ROBOCASA_DOF = len(STRETCH_ROBOCASA_MJCF_JOINT_NAMES)
_HELLO_STRETCH_DOF = HelloStretchIdx.HEAD_TILT + 1


def hello_stretch_config_from_joint_positions(
    q: np.ndarray,
    *,
    base_xyt: np.ndarray | None = None,
) -> np.ndarray:
    """Map simulator joint positions to the 11-DoF HelloStretchIdx layout.

    Robocasa Stretch MJCF publishes ``len(STRETCH_ROBOCASA_MJCF_JOINT_NAMES)`` arm/head joints
    (no planar base in ``q``); base pose comes from ``base_xyt`` / ZMQ ``base_pose``.
    """
    arr = np.asarray(q, dtype=np.float64).ravel()
    if arr.size == _HELLO_STRETCH_DOF:
        return arr.copy()
    if arr.size != _ROBOCASA_DOF:
        return arr

    out = np.zeros(_HELLO_STRETCH_DOF, dtype=np.float64)
    if base_xyt is not None:
        bp = np.asarray(base_xyt, dtype=np.float64).ravel()
        out[HelloStretchIdx.BASE_X] = float(bp[0])
        if bp.size > 1:
            out[HelloStretchIdx.BASE_Y] = float(bp[1])
        if bp.size > 2:
            out[HelloStretchIdx.BASE_THETA] = float(bp[2])

    out[HelloStretchIdx.LIFT] = arr[0]
    out[HelloStretchIdx.ARM] = float(np.sum(arr[1:5]))
    out[HelloStretchIdx.WRIST_YAW] = arr[5]
    out[HelloStretchIdx.WRIST_PITCH] = arr[6]
    out[HelloStretchIdx.WRIST_ROLL] = arr[7]
    out[HelloStretchIdx.HEAD_PAN] = arr[8]
    out[HelloStretchIdx.HEAD_TILT] = arr[9]
    return out


def robocasa_mjcf_joint_positions_from_hello_stretch(q: np.ndarray) -> np.ndarray | None:
    """Inverse of :func:`hello_stretch_config_from_joint_positions` for MJCF Rerun drive."""
    arr = np.asarray(q, dtype=np.float64).ravel()
    if arr.size == _ROBOCASA_DOF:
        return arr.copy()
    if arr.size != _HELLO_STRETCH_DOF:
        return None
    arm_q = float(arr[HelloStretchIdx.ARM]) / 4.0
    return np.array(
        [
            arr[HelloStretchIdx.LIFT],
            arm_q,
            arm_q,
            arm_q,
            arm_q,
            arr[HelloStretchIdx.WRIST_YAW],
            arr[HelloStretchIdx.WRIST_PITCH],
            arr[HelloStretchIdx.WRIST_ROLL],
            arr[HelloStretchIdx.HEAD_PAN],
            arr[HelloStretchIdx.HEAD_TILT],
        ],
        dtype=np.float64,
    )
