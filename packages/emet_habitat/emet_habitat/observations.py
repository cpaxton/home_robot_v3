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
* emet ``gps`` is planar ``(x, z)`` in Habitat horizontal axes.
* ``compass`` is a single yaw (radians, CCW positive).
* ``camera_pose`` is an OpenCV 4×4 camera-to-world matrix for depth unprojection into
  voxel-map world ``(X, Z, Y - floor_y)`` — see :mod:`emet.habitat.coordinates`.
"""

from __future__ import annotations

import numpy as np

from emet.core.interfaces import Observations
from emet.habitat.coordinates import habitat_agent_pose_from_state, habitat_observation_camera_pose


def habitat_rgb_depth_to_observations(
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    agent_state,
    intrinsics: np.ndarray,
    semantic: np.ndarray | None = None,
    sensor_rotation_offset: np.ndarray | None = None,
    floor_y: float | None = None,
    sensor_height: float = 1.5,
    sensor_uuid: str = "depth_sensor",
) -> Observations:
    """Build emet :class:`~emet.core.interfaces.Observations` from Habitat RGB-D.

    Args:
        rgb: ``H×W×3`` uint8 or float RGB image.
        depth: ``H×W`` depth in meters (may include a trailing singleton dim).
        agent_state: Habitat-Sim agent state with ``position`` and ``rotation``.
        intrinsics: ``3×3`` pinhole camera matrix ``K``.
        semantic: Optional ``H×W`` uint32 semantic instance id image.
        sensor_rotation_offset: Optional ``4×4`` transform applied to agent pose
            before building the camera pose (sensor mount offset).
        floor_y: Navmesh-snapped spawn Habitat ``Y`` for floor-relative obstacle height.
            Defaults to ``agent_state.position[1]`` when omitted.
        sensor_height: Body-frame sensor mount height when ``sensor_states`` unavailable.
        sensor_uuid: Habitat-Sim sensor uuid for ``agent_state.sensor_states``.

    Returns:
        Observations with ``gps``, ``compass``, ``rgb``, ``depth``, ``camera_K``,
        ``camera_pose``, and optional ``semantic``.
    """
    pos = np.asarray(agent_state.position, dtype=np.float64)
    hab_pose = habitat_agent_pose_from_state(agent_state)

    if sensor_rotation_offset is not None:
        hab_pose = hab_pose @ np.asarray(sensor_rotation_offset, dtype=np.float64)

    ref_floor_y = float(pos[1]) if floor_y is None else float(floor_y)
    camera_pose = habitat_observation_camera_pose(
        hab_pose,
        floor_y=ref_floor_y,
        agent_state=agent_state,
        sensor_uuid=sensor_uuid,
        sensor_height=sensor_height,
    )

    hab_R = hab_pose[:3, :3]
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
