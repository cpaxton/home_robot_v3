# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Map ScanNet simulator outputs to emet :class:`Observations`."""

from __future__ import annotations

import numpy as np

from emet.benchmarks.sqa3d.scannet.pose import camera_pose_opencv, gps_compass_from_pose
from emet.benchmarks.sqa3d.scannet.sens import scannet_camera_to_opencv_camera_to_world
from emet.core.interfaces import Observations


def scannet_rgb_depth_to_observations(
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    position: np.ndarray,
    quat_xyzw: np.ndarray,
    intrinsics: np.ndarray,
    sensor_height: float,
    camera_tilt_deg: float,
    camera_to_world: np.ndarray | None = None,
) -> Observations:
    gps, compass = gps_compass_from_pose(position, quat_xyzw)
    if camera_to_world is not None:
        camera_pose = scannet_camera_to_opencv_camera_to_world(camera_to_world)
    else:
        camera_pose = camera_pose_opencv(
            position,
            quat_xyzw,
            sensor_height=sensor_height,
            camera_tilt_deg=camera_tilt_deg,
        )

    if rgb.dtype != np.uint8:
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        rgb_u8 = rgb

    depth_m = np.asarray(depth, dtype=np.float32)
    if depth_m.ndim == 3:
        depth_m = depth_m[..., 0]
    # Open3D depth is view-space Z; treat as metric depth for emet fusion.
    depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0)

    return Observations(
        gps=gps,
        compass=compass,
        rgb=rgb_u8,
        depth=depth_m,
        semantic=None,
        camera_K=np.asarray(intrinsics, dtype=np.float64),
        camera_pose=camera_pose,
    )
