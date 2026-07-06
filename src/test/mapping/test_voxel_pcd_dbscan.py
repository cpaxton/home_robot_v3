# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Tests for DBSCAN pruning on navigation voxel PCD."""

import torch

from emet.mapping.voxel.voxel_dynamem import SparseVoxelMap


def test_sparse_voxel_map_stores_dbscan_min_samples():
    vm = SparseVoxelMap(voxel_pcd_dbscan_min_samples=8, log="test")
    assert vm._voxel_pcd_dbscan_min_samples == 8


def test_clear_points_dbscan_removes_small_out_of_view_cluster():
    """Points outside the current frustum survive the refresh pass; DBSCAN drops lone floaters."""
    from emet.utils.voxel import VoxelizedPointcloud

    vox = VoxelizedPointcloud(voxel_size=0.1)
    wall = torch.tensor(
        [
            [100.0, 0.0, 1.0],
            [100.15, 0.0, 1.0],
            [100.30, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    floater = torch.tensor([[100.0, 5.0, 1.0]], dtype=torch.float32)
    rgb = torch.zeros((4, 3), dtype=torch.float32)
    vox.add(torch.cat([wall, floater], dim=0), features=None, rgb=rgb, min_weight_per_voxel=1)

    depth = torch.ones((480, 640), dtype=torch.float32) * 2.0
    K = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    pose = torch.eye(4, dtype=torch.float32)

    vox.clear_points(depth, K, pose, min_samples_clear=3)
    assert vox._points is not None
    assert vox._points.shape[0] == 3
    assert not torch.any(torch.isclose(vox._points[:, 1], torch.tensor(5.0)))
