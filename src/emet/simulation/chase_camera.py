# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Third-person chase camera for MuJoCo sim recording.

``mjCAMERA_TRACKING`` pins lookat to the body origin (often floor-level on
``base_link``). With a behind/above orbit that line of sight cuts through the
torso mesh — the classic “camera inside the robot” look. This helper builds a
``mjCAMERA_FREE`` view that:

* looks at a point **above** the base (chest height)
* orbits behind the base ``+X`` axis (drive forward) by default
* follows base yaw each frame
"""

from __future__ import annotations

import os
from typing import Any

import mujoco
import numpy as np


def env_chase_distance() -> float:
    return float(os.environ.get("EMET_SIM_THIRD_PERSON_DISTANCE", "5.5"))


def env_chase_azimuth_offset() -> float:
    """Degrees added to base yaw for FREE-cam azimuth; ``0`` = directly behind (``-X``)."""
    return float(os.environ.get("EMET_SIM_THIRD_PERSON_AZIMUTH", "125"))


def env_chase_elevation() -> float:
    return float(os.environ.get("EMET_SIM_THIRD_PERSON_ELEVATION", "-28"))


def env_chase_lookat_z() -> float:
    """World-up offset from base origin to lookat (meters)."""
    return float(os.environ.get("EMET_SIM_THIRD_PERSON_LOOKAT_Z", "0.75"))


def base_yaw_deg(xmat_9: np.ndarray) -> float:
    """Yaw of body ``+X`` in the world XY plane (degrees)."""
    m = np.asarray(xmat_9, dtype=np.float64).reshape(3, 3)
    return float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))


def build_base_chase_camera(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    *,
    distance: float | None = None,
    azimuth_offset_deg: float | None = None,
    elevation_deg: float | None = None,
    lookat_z: float | None = None,
) -> mujoco.MjvCamera:
    """FREE chase camera looking at chest height above ``body_id``, from behind."""
    if body_id < 0:
        raise ValueError("body_id must be a valid MuJoCo body id")
    xpos = np.asarray(data.xpos[body_id], dtype=np.float64).reshape(3)
    yaw = base_yaw_deg(data.xmat[body_id])
    dist = max(0.8, float(distance if distance is not None else env_chase_distance()))
    az_off = float(azimuth_offset_deg if azimuth_offset_deg is not None else env_chase_azimuth_offset())
    elev = float(elevation_deg if elevation_deg is not None else env_chase_elevation())
    look_z = float(lookat_z if lookat_z is not None else env_chase_lookat_z())

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = xpos + np.array([0.0, 0.0, look_z], dtype=np.float64)
    cam.distance = dist
    # Azimuth 0 places the camera on world -X when yaw=0 → behind body +X.
    cam.azimuth = yaw + az_off
    cam.elevation = elev
    return cam


def apply_chase_frustum_near(scene: Any, near: float = 0.05) -> None:
    """Optionally lower the near plane without changing FOV (scale aperture with near).

    Large-kitchen ``stat.extent`` can make ``znear * extent`` slice nearby meshes. Setting
    ``frustum_near`` alone (without scaling top/bottom/width) shrinks the FOV and looks like
    an extreme zoom into the robot — do not do that.
    """
    near = float(near)
    ncam = int(getattr(scene, "ncamera", 2) or 2)
    for i in range(max(0, ncam)):
        try:
            cam = scene.camera[i]
        except (IndexError, AttributeError):
            break
        old_near = float(cam.frustum_near)
        if old_near <= 1e-9 or near >= old_near:
            # Already close enough (or would push the plane farther) — leave alone.
            continue
        scale = near / old_near
        cam.frustum_near = near
        cam.frustum_bottom = float(cam.frustum_bottom) * scale
        cam.frustum_top = float(cam.frustum_top) * scale
        cam.frustum_width = float(cam.frustum_width) * scale
