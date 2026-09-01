# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tests for explored/visited stamping pose selection in SparseVoxelMap."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import torch

from emet.mapping.voxel.voxel_dynamem import SparseVoxelMap


def _make_map(**kwargs) -> SparseVoxelMap:
    defaults = {
        "resolution": 0.05,
        "semantic_memory_resolution": 0.05,
        "feature_dim": 3,
        "use_instance_memory": False,
        "encoder": None,
        "device": "cpu",
        "map_2d_device": "cpu",
        "add_local_radius_points": True,
        "local_radius": 0.25,
    }
    defaults.update(kwargs)
    return SparseVoxelMap(**defaults)


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


def test_every_observation_stamps_obstacles_from_a_stationary_base() -> None:
    """A rotate-in-place scan is the main way the agent maps a room, so it must stamp."""
    voxel_map = _make_map()
    base_pose = torch.tensor([1.31, -3.47, 0.0], dtype=torch.float32)

    with patch.object(voxel_map.voxel_pcd, "add", autospec=True) as mock_add:
        _add_observation(voxel_map, base_pose=base_pose)
        _add_observation(voxel_map, base_pose=base_pose)

    assert mock_add.call_count == 2


def test_every_observation_stamps_obstacles_after_base_moves() -> None:
    voxel_map = _make_map()

    with patch.object(voxel_map.voxel_pcd, "add", autospec=True) as mock_add:
        _add_observation(voxel_map, base_pose=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
        _add_observation(voxel_map, base_pose=torch.tensor([0.5, 0.0, 0.0], dtype=torch.float32))

    assert mock_add.call_count == 2


def test_visited_stamped_only_on_first_observation() -> None:
    """Rotate-in-place must not grow a visited trail that paints over coverage holes."""
    voxel_map = _make_map()
    first = torch.tensor([1.31, -3.47, 0.0], dtype=torch.float32)
    second = torch.tensor([2.31, -3.47, 0.0], dtype=torch.float32)
    _add_observation(voxel_map, base_pose=first)
    _add_observation(voxel_map, base_pose=second)

    visited = voxel_map._visited.detach().cpu().numpy()
    res = float(voxel_map.grid_resolution)
    origin = voxel_map.grid_origin[:2].detach().cpu().numpy()
    first_ij = ((first[:2].numpy() / res) + origin).astype(int)
    second_ij = ((second[:2].numpy() / res) + origin).astype(int)

    assert visited[first_ij[0], first_ij[1]] > 0
    assert visited[second_ij[0], second_ij[1]] == 0


def test_visited_every_step_when_configured() -> None:
    voxel_map = _make_map(add_local_radius_every_step=True)
    first = torch.tensor([1.31, -3.47, 0.0], dtype=torch.float32)
    second = torch.tensor([2.31, -3.47, 0.0], dtype=torch.float32)
    _add_observation(voxel_map, base_pose=first)
    _add_observation(voxel_map, base_pose=second)

    visited = voxel_map._visited.detach().cpu().numpy()
    res = float(voxel_map.grid_resolution)
    origin = voxel_map.grid_origin[:2].detach().cpu().numpy()
    second_ij = ((second[:2].numpy() / res) + origin).astype(int)
    assert visited[second_ij[0], second_ij[1]] > 0


def test_explored_keeps_one_cell_gap_between_islands() -> None:
    """Morphological close used to merge nearby coverage islands into a solid carpet."""
    voxel_map = _make_map(add_local_radius_points=False, smooth_kernel_size=3)
    visited = voxel_map._visited
    i = int(visited.shape[0] // 2)
    j = int(visited.shape[1] // 2)
    visited[i, j] = 1
    visited[i, j + 2] = 1
    _, explored = voxel_map.get_2d_map()
    exp = explored.detach().cpu().numpy()
    assert exp[i, j]
    assert exp[i, j + 2]
    assert not exp[i, j + 1]
