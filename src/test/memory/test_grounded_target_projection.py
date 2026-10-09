# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from itertools import product

import numpy as np
import pytest

from emet.memory.grounded_target import GroundedTarget


def target_in_camera(pose, lo=(-0.1, -0.2, 1), hi=(0.1, 0.2, 2)):
    corners = np.asarray(list(product(*zip(lo, hi, strict=True))))
    points = np.tile(corners @ pose[:3, :3].T + pose[:3, 3], (2, 1))
    return GroundedTarget(1, 2, 3, points)


K = np.array([[100, 0, 50], [0, 100, 40], [0, 0, 1]])


def test_projection_uses_camera_to_world_pose_and_all_bounds_corners():
    pose = np.array([[0, -1, 0, 3], [0, 0, -1, -2], [1, 0, 0, 1], [0, 0, 0, 1]])
    target = target_in_camera(pose)
    np.testing.assert_allclose(target.project_box(K, pose, (80, 100)), [40, 20, 60, 60])


def test_projection_clips_partial_extent_but_rejects_outside_and_camera_plane():
    pose = np.eye(4)
    target = target_in_camera(pose, lo=(-1, -1, 1), hi=(0.1, 0.2, 2))
    np.testing.assert_allclose(target.project_box(K, pose, (80, 100)), [0, 0, 60, 60])
    for lo, hi in [((2, 2, 1), (3, 3, 2)), ((0, 0, -1), (1, 1, 1)), ((0, 0, -2), (1, 1, -1))]:
        with pytest.raises(ValueError):
            target_in_camera(pose, lo, hi).project_box(K, pose, (80, 100))


@pytest.mark.parametrize(
    "kind", ["missing_K", "missing_pose", "nan", "reflection", "scaled_pose", "negative_focal", "empty_image"]
)
def test_projection_fails_closed_on_invalid_calibration(kind):
    pose, intrinsic, shape = np.eye(4), K.copy(), (80, 100)
    target = target_in_camera(pose)
    if kind == "missing_K":
        intrinsic = None
    elif kind == "missing_pose":
        pose = None
    elif kind == "nan":
        pose[0, 0] = np.nan
    elif kind == "reflection":
        pose[0, 0] = -1
    elif kind == "scaled_pose":
        pose[0, 0] = 2
    elif kind == "negative_focal":
        intrinsic[0, 0] = -100
    else:
        shape = (0, 100)
    with pytest.raises(ValueError):
        target.project_box(intrinsic, pose, shape)
