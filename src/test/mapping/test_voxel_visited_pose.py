# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tests for explored/visited stamping pose selection in SparseVoxelMap."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import torch

from emet.mapping.voxel.voxel_dynamem import SparseVoxelMap


def _make_map() -> SparseVoxelMap:
    return SparseVoxelMap(
        resolution=0.05,
        semantic_memory_resolution=0.05,
        feature_dim=3,
        use_instance_memory=False,
        encoder=None,
        device="cpu",
        map_2d_device="cpu",
        add_local_radius_points=True,
    )


def _add_observation(
    voxel_map: SparseVoxelMap,
    *,
    base_pose: torch.Tensor | None,
) -> None:
    camera_pose = torch.eye(4, dtype=torch.float32)
    camera_pose[0, 3] = -1.31
    camera_pose[1, 3] = -3.47
    camera_pose[2, 3] = -2.79
    rgb = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    xyz = torch.tensor([[1.0, 0.5, -3.0]], dtype=torch.float32)
    voxel_map.add(
        camera_pose,
        rgb,
        xyz=xyz,
        xyz_frame="world",
        base_pose=base_pose,
    )


def test_update_visited_uses_base_pose_when_present() -> None:
    voxel_map = _make_map()
    base_pose = torch.tensor([1.31, -3.47, 0.0], dtype=torch.float32)

    with patch.object(voxel_map, "_update_visited", autospec=True) as mock_update:
        _add_observation(voxel_map, base_pose=base_pose)

    mock_update.assert_called_once()
    pose = mock_update.call_args[0][0]
    np.testing.assert_allclose(pose[:2].detach().cpu().numpy(), base_pose[:2].numpy())


def test_single_visited_stamp_at_base_grid() -> None:
    """Habitat-style poses: visited disk should center on base_pose grid, not camera translation."""
    voxel_map = _make_map()
    base_pose = torch.tensor([1.31, -3.47, 0.0], dtype=torch.float32)
    _add_observation(voxel_map, base_pose=base_pose)

    visited = voxel_map._visited.detach().cpu().numpy()
    assert int(visited.sum()) > 0

    res = float(voxel_map.grid_resolution)
    origin = voxel_map.grid_origin[:2].detach().cpu().numpy()
    base_ij = ((base_pose[:2].numpy() / res) + origin).astype(int)
    camera_pose = torch.eye(4, dtype=torch.float32)
    camera_pose[0, 3] = -1.31
    camera_pose[1, 3] = -3.47
    cam_ij = ((camera_pose[:3, 3][:2].numpy() / res) + origin).astype(int)

    assert visited[base_ij[0], base_ij[1]] > 0
    assert visited[cam_ij[0], cam_ij[1]] == 0


def test_update_visited_falls_back_to_camera_without_base_pose() -> None:
    voxel_map = _make_map()
    camera_xy = torch.tensor([-1.31, -3.47], dtype=torch.float32)

    with patch.object(voxel_map, "_update_visited", autospec=True) as mock_update:
        _add_observation(voxel_map, base_pose=None)

    mock_update.assert_called_once()
    pose = mock_update.call_args[0][0]
    np.testing.assert_allclose(pose[:2].detach().cpu().numpy(), camera_xy.numpy())
