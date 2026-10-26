# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""On-demand binding to the robot's already-running measured RGBD stream."""

import numpy as np


def capture_arrays(observation):
    """Unproject the same calibrated sensor depth used by the mapping pipeline.

    No simulator scene metadata, estimated-depth fallback, inference call, or
    client-side timestamp replacement is used. Camera pose is expected in the
    same world frame as the reference processor's points.
    """
    from emet.utils.image import Camera, camera_xyz_to_global_xyz

    rgb, depth = np.asarray(observation.rgb), np.asarray(observation.depth)
    intrinsic, pose = np.asarray(observation.camera_K, dtype=float), np.asarray(observation.camera_pose, dtype=float)
    if (
        depth.ndim != 2
        or min(depth.shape) == 0
        or rgb.shape != (*depth.shape, 3)
        or rgb.dtype != np.uint8
        or intrinsic.shape != (3, 3)
        or pose.shape != (4, 4)
        or not np.isfinite(intrinsic).all()
        or not np.isfinite(pose).all()
        or intrinsic[0, 0] <= 0
        or intrinsic[1, 1] <= 0
        or not np.allclose(intrinsic[2], [0, 0, 1])
        or not np.allclose([intrinsic[0, 1], intrinsic[1, 0]], 0.0)
        or not np.allclose(pose[3], [0, 0, 0, 1])
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-5)
    ):
        raise ValueError("invalid_observed_camera_geometry")
    camera = Camera.from_K(intrinsic, width=depth.shape[1], height=depth.shape[0])
    world = camera_xyz_to_global_xyz(camera.depth_to_xyz(depth), pose)
    return rgb, depth, world


def bind_observed_capture_source(provider, robot, inputs):
    """Only the concrete RGBD provider owns this binding; other providers opt in separately."""
    from .observed_discovery import RgbdTaskProvider

    if inputs == "observed" and isinstance(provider, RgbdTaskProvider):
        provider.bind_observation_source(robot.get_observation)
        return True
    return False
