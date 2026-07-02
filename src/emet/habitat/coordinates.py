# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Habitat Y-up world ↔ emet voxel-map world coordinates for HM-EQA mapping.

Emet :class:`~emet.mapping.voxel.voxel_dynamem.SparseVoxelMap` expects world points as
``(planar_x, planar_y, height)`` with ``obs_min_height`` / ``obs_max_height`` applied on
the height axis. For Habitat episodes:

* planar axes match ``Observations.gps`` = Habitat ``(X, Z)``
* height = Habitat ``Y - floor_y`` (navmesh-snapped spawn Y)
"""

from __future__ import annotations

import numpy as np

# Habitat-Sim / OpenGL camera (+X right, +Y up, -Z forward) → OpenCV (+X right, +Y down, +Z forward).
_GL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


def _rotation_matrix_from_agent_rotation(rot) -> np.ndarray:
    """Convert Habitat-Sim agent rotation to a 3×3 rotation matrix."""
    if not hasattr(rot, "shape"):
        try:
            import quaternion as npq
        except ImportError as err:
            raise ImportError("numpy-quaternion is required for Habitat agent rotations") from err
        if isinstance(rot, npq.quaternion):
            return npq.as_rotation_matrix(rot).astype(np.float64)

    try:
        arr = np.asarray(rot, dtype=np.float64)
    except (TypeError, ValueError):
        import quaternion as npq

        if isinstance(rot, npq.quaternion):
            return npq.as_rotation_matrix(rot).astype(np.float64)
        raise

    if arr.shape == (3, 3):
        return arr
    import quaternion as npq

    if isinstance(rot, npq.quaternion):
        return npq.as_rotation_matrix(rot).astype(np.float64)
    coeffs = arr.reshape(-1)
    if coeffs.shape[0] == 4:
        q = npq.quaternion(coeffs[0], coeffs[1], coeffs[2], coeffs[3])
        return npq.as_rotation_matrix(q).astype(np.float64)
    return coeffs.reshape(3, 3)


def _pose_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    pose[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return pose


def habitat_yup_permutation_matrix() -> np.ndarray:
    """4×4 matrix mapping Habitat ``(X, Y, Z)`` → voxel ``(X, Z, Y)`` (no floor offset)."""
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def habitat_to_voxel_world_transform(floor_y: float = 0.0) -> np.ndarray:
    """Affine map Habitat Y-up world → emet voxel world with floor-relative height."""
    t = habitat_yup_permutation_matrix().copy()
    t[2, 3] = -float(floor_y)
    return t


def habitat_world_to_voxel_xyz(points: np.ndarray, *, floor_y: float = 0.0) -> np.ndarray:
    """Map N×3 Habitat world points to voxel-map world coordinates."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    hom = np.hstack([pts, ones])
    return (hom @ habitat_to_voxel_world_transform(floor_y).T)[:, :3]


def opengl_camera_to_opencv_camera_to_world(c2w_gl: np.ndarray) -> np.ndarray:
    """OpenGL camera-to-world (Habitat Y-up) → OpenCV camera-to-world (same world frame)."""
    return np.asarray(c2w_gl, dtype=np.float64) @ _GL_TO_OPENCV


def _sensor_pose_habitat(agent_state, *, sensor_uuid: str = "depth_sensor") -> np.ndarray | None:
    sensor_states = getattr(agent_state, "sensor_states", None)
    if not sensor_states or sensor_uuid not in sensor_states:
        return None
    sensor = sensor_states[sensor_uuid]
    rot = _rotation_matrix_from_agent_rotation(sensor.rotation)
    pos = np.asarray(sensor.position, dtype=np.float64).reshape(3)
    return _pose_from_rotation_translation(rot, pos)


def habitat_agent_to_opencv_camera_pose(
    hab_pose: np.ndarray,
    *,
    agent_state=None,
    sensor_uuid: str = "depth_sensor",
    sensor_height: float = 1.5,
    camera_tilt_deg: float = -30.0,
) -> np.ndarray:
    """OpenCV camera-to-world in Habitat Y-up (no Stretch z-up flip).

    Prefers Habitat-Sim ``agent_state.sensor_states`` when present; otherwise uses
    ``hab_pose`` (agent body) with a vertical sensor mount offset and optional pitch.
    """
    sensor_gl = _sensor_pose_habitat(agent_state, sensor_uuid=sensor_uuid) if agent_state is not None else None
    if sensor_gl is None:
        mount = np.eye(4, dtype=np.float64)
        mount[1, 3] = float(sensor_height)
        pitch = np.deg2rad(float(camera_tilt_deg))
        cp, sp = float(np.cos(pitch)), float(np.sin(pitch))
        mount[:3, :3] = np.array(
            [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]],
            dtype=np.float64,
        )
        sensor_gl = np.asarray(hab_pose, dtype=np.float64) @ mount
    return opengl_camera_to_opencv_camera_to_world(sensor_gl)


def habitat_observation_camera_pose(
    hab_pose: np.ndarray,
    *,
    floor_y: float = 0.0,
    agent_state=None,
    sensor_uuid: str = "depth_sensor",
    sensor_height: float = 1.5,
    camera_tilt_deg: float = -30.0,
) -> np.ndarray:
    """OpenCV camera-to-voxel-world for :class:`~emet.core.interfaces.Observations`."""
    cam_hab = habitat_agent_to_opencv_camera_pose(
        hab_pose,
        agent_state=agent_state,
        sensor_uuid=sensor_uuid,
        sensor_height=sensor_height,
        camera_tilt_deg=camera_tilt_deg,
    )
    return habitat_to_voxel_world_transform(floor_y) @ cam_hab


def habitat_agent_pose_from_state(agent_state) -> np.ndarray:
    """Build 4×4 agent body pose in Habitat Y-up from a Habitat-Sim agent state."""
    pos = np.asarray(agent_state.position, dtype=np.float64).reshape(3)
    rot = _rotation_matrix_from_agent_rotation(agent_state.rotation)
    return _pose_from_rotation_translation(rot, pos)
