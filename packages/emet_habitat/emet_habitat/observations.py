# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Map Habitat-Sim sensor outputs to emet :class:`~emet.core.interfaces.Observations`.

Coordinate conventions:

* Habitat agent state uses Y-up world coordinates.
* emet ``gps`` is planar ``(x, z)`` with **x forward, y left** (Stretch nav convention).
* ``compass`` is a single yaw (radians, CCW positive).
* ``camera_pose`` is an OpenCV-style 4×4 camera-to-world matrix for depth unprojection.
"""

from __future__ import annotations

import numpy as np

from emet.core.interfaces import Observations
from emet.utils.pose import convert_pose_habitat_to_opencv


def _agent_rotation_matrix(rot) -> np.ndarray:
    """Convert Habitat-Sim agent rotation to a 3×3 rotation matrix.

    Args:
        rot: ``quaternion.quaternion`` or length-4 array ``(w, x, y, z)``.

    Returns:
        Rotation matrix ``R`` such that ``R @ v`` maps body vectors to world.
    """
    import quaternion as npq

    if isinstance(rot, npq.quaternion):
        return npq.as_rotation_matrix(rot).astype(np.float64)
    coeffs = np.asarray(rot, dtype=np.float64).reshape(-1)
    if coeffs.shape[0] == 4:
        q = npq.quaternion(coeffs[0], coeffs[1], coeffs[2], coeffs[3])
        return npq.as_rotation_matrix(q).astype(np.float64)
    return coeffs.reshape(3, 3)


def habitat_rgb_depth_to_observations(
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    agent_state,
    intrinsics: np.ndarray,
    semantic: np.ndarray | None = None,
    sensor_rotation_offset: np.ndarray | None = None,
) -> Observations:
    """Build emet :class:`~emet.core.interfaces.Observations` from Habitat RGB-D.

    Args:
        rgb: ``H×W×3`` uint8 or float RGB image.
        depth: ``H×W`` depth in meters (may include a trailing singleton dim).
        agent_state: Habitat-Sim agent state with ``position`` and ``rotation``.
        intrinsics: ``3×3`` pinhole camera matrix ``K``.
        semantic: Optional ``H×W`` uint32 semantic instance id image.
        sensor_rotation_offset: Optional ``4×4`` transform applied to agent pose
            before converting to OpenCV camera frame (sensor mount offset).

    Returns:
        Observations with ``gps``, ``compass``, ``rgb``, ``depth``, ``camera_K``,
        ``camera_pose``, and optional ``semantic``.
    """
    pos = np.asarray(agent_state.position, dtype=np.float64)
    rot = agent_state.rotation

    hab_R = _agent_rotation_matrix(rot)
    hab_pose = np.eye(4, dtype=np.float64)
    hab_pose[:3, :3] = hab_R
    hab_pose[:3, 3] = pos

    if sensor_rotation_offset is not None:
        hab_pose = hab_pose @ np.asarray(sensor_rotation_offset, dtype=np.float64)

    camera_pose = convert_pose_habitat_to_opencv(hab_pose)

    forward = hab_R[:, 2]
    heading = float(np.arctan2(forward[0], forward[2]))
    gps = np.array([float(pos[0]), float(pos[2])], dtype=np.float64)
    compass = np.array([heading], dtype=np.float64)

    if rgb.dtype != np.uint8:
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        rgb_u8 = rgb

    depth_m = np.asarray(depth, dtype=np.float32)
    if depth_m.ndim == 3:
        depth_m = depth_m[..., 0]

    sem_u32 = None
    if semantic is not None:
        sem_u32 = np.asarray(semantic, dtype=np.uint32)
        if sem_u32.ndim == 3:
            sem_u32 = sem_u32[..., 0]

    return Observations(
        gps=gps,
        compass=compass,
        rgb=rgb_u8,
        depth=depth_m,
        semantic=sem_u32,
        camera_K=np.asarray(intrinsics, dtype=np.float64),
        camera_pose=camera_pose,
    )
