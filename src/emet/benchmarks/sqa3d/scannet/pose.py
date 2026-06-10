# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""SQA3D agent pose in ScanNet aligned coordinates (Z-up, planarity in XY)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

# Agent body frame: +X forward, +Y left, +Z up (matches SQA3D localization yaw about Z).
_AGENT_FORWARD_LOCAL = np.array([1.0, 0.0, 0.0], dtype=np.float64)
_AGENT_UP_LOCAL = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def quat_xyzw_to_rotation(quat_xyzw: tuple[float, float, float, float] | np.ndarray) -> R:
    return R.from_quat(np.asarray(quat_xyzw, dtype=np.float64))


def agent_pose_matrix(
    position: tuple[float, float, float] | np.ndarray,
    quat_xyzw: tuple[float, float, float, float] | np.ndarray,
) -> np.ndarray:
    """4x4 body-to-world transform."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_rotation(quat_xyzw).as_matrix()
    T[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return T


def planar_heading_rad(rotation: R) -> float:
    forward = rotation.apply(_AGENT_FORWARD_LOCAL)
    forward[2] = 0.0
    norm = float(np.linalg.norm(forward[:2]))
    if norm < 1e-8:
        return 0.0
    forward[:2] /= norm
    return float(np.arctan2(forward[1], forward[0]))


def gps_compass_from_pose(
    position: tuple[float, float, float] | np.ndarray,
    quat_xyzw: tuple[float, float, float, float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """emet ``gps`` (x,y) and ``compass`` (yaw) from ScanNet XY plane."""
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    rot = quat_xyzw_to_rotation(quat_xyzw)
    gps = np.array([pos[0], pos[1]], dtype=np.float64)
    compass = np.array([planar_heading_rad(rot)], dtype=np.float64)
    return gps, compass


def camera_pose_opencv(
    position: tuple[float, float, float] | np.ndarray,
    quat_xyzw: tuple[float, float, float, float] | np.ndarray,
    *,
    sensor_height: float = 1.5,
    camera_tilt_deg: float = -30.0,
) -> np.ndarray:
    """World-to-camera 4x4 in OpenCV convention for emet ``Observations``."""
    from emet.utils.pose import convert_pose_habitat_to_opencv

    body = agent_pose_matrix(position, quat_xyzw)
    cam_offset = np.eye(4, dtype=np.float64)
    cam_offset[2, 3] = float(sensor_height)
    tilt = np.eye(4, dtype=np.float64)
    tilt[:3, :3] = R.from_euler("y", np.deg2rad(camera_tilt_deg)).as_matrix()
    world_cam = body @ cam_offset @ tilt
    return convert_pose_habitat_to_opencv(world_cam)


def apply_turn(
    quat_xyzw: np.ndarray,
    delta_yaw_rad: float,
) -> np.ndarray:
    """Rotate agent about world +Z."""
    delta = R.from_euler("z", delta_yaw_rad)
    new_rot = delta * quat_xyzw_to_rotation(quat_xyzw)
    return new_rot.as_quat()


def apply_forward(
    position: np.ndarray,
    quat_xyzw: np.ndarray,
    distance_m: float,
) -> np.ndarray:
    """Translate agent in the XY plane along body forward."""
    rot = quat_xyzw_to_rotation(quat_xyzw)
    fwd = rot.apply(_AGENT_FORWARD_LOCAL)
    fwd[2] = 0.0
    norm = float(np.linalg.norm(fwd[:2]))
    if norm > 1e-8:
        fwd[:2] /= norm
    pos = np.asarray(position, dtype=np.float64).copy()
    pos[0] += fwd[0] * distance_m
    pos[1] += fwd[1] * distance_m
    return pos
