# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import numpy as np
import pytest

from emet.mapping.rgbd_odometry import RGBDOdometry, registration_quality


def frame():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (64, 80, 3), dtype=np.uint8), np.ones((64, 80), dtype=np.float32)


def calibration():
    return np.array([[60, 0, 40], [0, 60, 32], [0, 0, 1]], dtype=float)


def test_registration_quality_requires_geometry_and_appearance():
    rgb, depth = frame()
    assert registration_quality(rgb, depth, rgb, depth, calibration(), np.eye(4)) == (1.0, 0.0, 0.0)
    overlap, _, _ = registration_quality(rgb, depth, rgb, depth + 1, calibration(), np.eye(4))
    assert overlap == 0
    overlap, _, color_error = registration_quality(rgb, depth, 255 - rgb, depth, calibration(), np.eye(4))
    assert overlap == 1 and color_error > 0.15


def test_blocked_motion_does_not_become_commanded_progress():
    rgb, depth = frame()
    odometry = RGBDOdometry(calibration(), np.eye(4))
    assert odometry.update(rgb, depth, 1).ok
    guess = np.eye(4)
    guess[0, 3] = -0.1  # Forward prediction despite unchanged measured RGB-D.
    result = odometry.update(rgb, depth, 2, source_to_target_guess=guess)
    assert result.ok and result.status == "tracked"
    assert np.linalg.norm(result.camera_pose[:3, 3]) < 0.003
    assert np.arccos(np.clip((np.trace(result.camera_pose[:3, :3]) - 1) / 2, -1, 1)) < 0.002
    assert odometry.timestamp_ns == 2


def test_tracking_loss_keeps_last_accepted_pose_and_requires_reset():
    rgb, depth = frame()
    odometry = RGBDOdometry(calibration(), np.eye(4))
    odometry.update(rgb, depth, 1)
    result = odometry.update(rgb, np.zeros_like(depth), 2)
    assert not result.ok and result.camera_pose is None
    np.testing.assert_array_equal(odometry.pose, np.eye(4))
    assert odometry.timestamp_ns == 1
    assert odometry.update(rgb, depth, 3).status == "tracking_lost"


def test_stale_frames_and_changed_calibration_are_rejected():
    rgb, depth = frame()
    odometry = RGBDOdometry(calibration(), np.eye(4))
    odometry.update(rgb, depth, 1)
    with pytest.raises(ValueError, match="fresh timestamp"):
        odometry.update(rgb, depth, 1)
    with pytest.raises(ValueError, match="dimensions"):
        odometry.update(rgb[:32], depth[:32], 2)
    with pytest.raises(ValueError, match="right-handed"):
        RGBDOdometry(calibration(), np.diag([-1, 1, 1, 1]))
