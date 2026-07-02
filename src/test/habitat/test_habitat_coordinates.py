# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for Habitat → voxel-map coordinate transforms and observation poses."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

from emet.habitat.coordinates import (
    habitat_agent_pose_from_state,
    habitat_observation_camera_pose,
    habitat_to_voxel_world_transform,
    habitat_world_to_voxel_xyz,
)
from emet.utils.point_cloud_torch import unproject_masked_depth_to_xyz_coordinates


def _agent_state(
    position: tuple[float, float, float],
    *,
    heading: float = 0.0,
    camera_tilt_deg: float = -30.0,
    sensor_height: float = 1.5,
) -> SimpleNamespace:
    """Minimal Habitat-like agent state with yaw-only body and pitched sensor."""
    q_yaw = R.from_rotvec([0.0, float(heading), 0.0])
    body_rot = q_yaw.as_matrix()
    q_tilt = R.from_rotvec([np.deg2rad(camera_tilt_deg), 0.0, 0.0])
    sensor_rot = (q_yaw * q_tilt).as_matrix()
    body_pos = np.array(position, dtype=np.float32)
    sensor_pos = body_pos + body_rot @ np.array([0.0, sensor_height, 0.0], dtype=np.float64)
    sensor = SimpleNamespace(
        rotation=sensor_rot.astype(np.float32),
        position=sensor_pos.astype(np.float32),
    )
    return SimpleNamespace(
        position=body_pos,
        rotation=body_rot.astype(np.float32),
        sensor_states={"depth_sensor": sensor, "color_sensor": sensor},
    )


def test_habitat_world_to_voxel_xyz_permutation_and_floor() -> None:
    pts = np.array(
        [
            [1.31, -2.79, -3.47],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    out = habitat_world_to_voxel_xyz(pts, floor_y=-2.79)
    np.testing.assert_allclose(out[0], [1.31, -3.47, 0.0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(out[1], [0.0, 0.0, 2.79], rtol=0, atol=1e-6)


def test_habitat_to_voxel_world_transform_is_4x4() -> None:
    t = habitat_to_voxel_world_transform(floor_y=-1.0)
    assert t.shape == (4, 4)
    np.testing.assert_allclose(t[2, 3], 1.0, atol=1e-9)


def test_unprojected_centroid_xz_matches_gps_within_tolerance() -> None:
    """00006-style spawn: planar PCD X must align with gps (no OpenCV world X flip)."""
    pos = (-4.55, 0.0, 3.30)
    floor_y = 0.0
    agent = _agent_state(pos, heading=0.0, camera_tilt_deg=-30.0)
    hab_pose = habitat_agent_pose_from_state(agent)
    camera_pose = habitat_observation_camera_pose(
        hab_pose,
        floor_y=floor_y,
        agent_state=agent,
        sensor_height=1.5,
    )

    h, w = 48, 64
    fx = fy = 320.0
    cx, cy = w / 2.0, h / 2.0
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    depth = np.full((h, w), 1.2, dtype=np.float32)

    depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
    pose_t = torch.from_numpy(camera_pose).unsqueeze(0).float()
    inv_k = torch.linalg.inv(torch.from_numpy(intrinsics[:3, :3]).unsqueeze(0).float())
    world_xyz = unproject_masked_depth_to_xyz_coordinates(depth=depth_t, pose=pose_t, inv_intrinsics=inv_k)
    centroid_planar = world_xyz[:, :2].mean(dim=0).detach().cpu().numpy()

    gps_xz = np.array([pos[0], pos[2]], dtype=np.float64)
    assert np.sign(centroid_planar[0]) == np.sign(gps_xz[0]), "X sign must match gps (no OpenCV world flip)"
    assert abs(centroid_planar[0] - gps_xz[0]) < 0.5, f"PCD X {centroid_planar[0]} vs gps X {gps_xz[0]}"
    # Forward depth places centroid ahead of base in +Z for heading 0 (not co-located with gps Z).
    assert centroid_planar[1] < gps_xz[1], "tilted forward depth should centroid below base Z at heading 0"


def test_floor_relative_height_in_obstacle_band() -> None:
    """00025-style spawn Y≈-2.79: wall points should land in [0.2, 1.5] m height axis."""
    pos = (1.31, -2.79, -3.47)
    floor_y = -2.79
    agent = _agent_state(pos, heading=0.0, camera_tilt_deg=-30.0)
    hab_pose = habitat_agent_pose_from_state(agent)
    camera_pose = habitat_observation_camera_pose(
        hab_pose,
        floor_y=floor_y,
        agent_state=agent,
        sensor_height=1.5,
    )

    h, w = 48, 64
    fx = fy = 320.0
    intrinsics = np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=np.float64)
    depth = np.full((h, w), 1.2, dtype=np.float32)

    depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
    pose_t = torch.from_numpy(camera_pose).unsqueeze(0).float()
    inv_k = torch.linalg.inv(torch.from_numpy(intrinsics[:3, :3]).unsqueeze(0).float())
    world_xyz = unproject_masked_depth_to_xyz_coordinates(depth=depth_t, pose=pose_t, inv_intrinsics=inv_k)
    heights = world_xyz[:, 2].detach().cpu().numpy()
    median_h = float(np.median(heights))
    assert 0.2 <= median_h <= 1.5, f"median height axis-2 {median_h:.2f} m outside obstacle band"


def test_permutation_matrix_without_habitat_sim() -> None:
    """Pure numpy checks run in the main venv without habitat-sim."""
    pt = np.array([[-4.55, 0.0, 3.30]], dtype=np.float64)
    out = habitat_world_to_voxel_xyz(pt, floor_y=0.0)
    np.testing.assert_allclose(out[0], [-4.55, 3.30, 0.0], rtol=0, atol=1e-6)


def test_legacy_opencv_world_flip_would_invert_x() -> None:
    """Document regression: Stretch world flip inverts Habitat X vs gps."""
    from emet.utils.pose import convert_pose_habitat_to_opencv

    hab_pose = np.eye(4, dtype=np.float64)
    hab_pose[0, 3] = -4.55
    hab_pose[1, 3] = 0.0
    hab_pose[2, 3] = 3.30
    legacy = convert_pose_habitat_to_opencv(hab_pose.copy())
    assert legacy[0, 3] > 0.0
    assert hab_pose[0, 3] < 0.0


def test_no_opencv_x_flip_on_planar_axes() -> None:
    """Voxel planar X must keep Habitat sign (00006 regression)."""
    pt = np.array([[-4.55, 1.0, 3.30]], dtype=np.float64)
    out = habitat_world_to_voxel_xyz(pt, floor_y=0.0)
    assert out[0, 0] < 0.0
    assert out[0, 1] == pytest.approx(3.30)
