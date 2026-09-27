# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
import numpy as np
import torch

from emet.mapping.voxel.voxel_dynamem import SparseVoxelMap
from emet.mapping.voxel.voxel_map_dynamem import SparseVoxelMapNavigationSpace
from emet.motion.algo.a_star import AStar
from emet.robots.footprint import Footprint


def test_fractional_start_repair_preserves_unknown_and_obstacle_guards():
    from types import SimpleNamespace

    obstacles = torch.zeros((1024, 1024), dtype=torch.bool)
    explored = torch.ones_like(obstacles)
    for cell in [(498, 507), (498, 508), (498, 510), (498, 511)]:
        explored[cell] = False
    original = explored.clone()
    space = object.__new__(SparseVoxelMapNavigationSpace)
    space.obstacle_map_mode = "physical"
    space._footprint = Footprint(length=0.33, width=0.34, length_offset=-0.1)
    space.voxel_map = SimpleNamespace(
        grid_resolution=0.1,
        grid=SimpleNamespace(grid_origin=torch.tensor([512, 512, 0])),
        get_navigation_map=lambda: (obstacles, explored),
    )
    pose = np.array([-1.0023702383, -0.2798024416, 0.0185558926])
    assert space.is_valid(pose)
    assert torch.equal(explored, original)
    cells = space._footprint.grid_cells(0.1, pose, [512, 512])
    cell = tuple(cells[0])
    explored[cell] = False
    assert not space.is_valid(pose)
    assert space.last_validity["unknown_footprint_cells"] == 1
    assert space.last_validity["reason"] == "unobserved_footprint"
    obstacles[cell] = True
    assert not space.is_valid(pose)
    assert space.last_validity["reason"] == "occupied_footprint"


def test_physical_map_keeps_raw_geometry_and_shared_cache(tmp_path):
    vm = SparseVoxelMap(
        grid_size=[40, 40],
        grid_resolution=0.1,
        pad_obstacles=2,
        smooth_kernel_size=0,
        obs_min_density=0.1,
        log=str(tmp_path),
        use_instance_memory=False,
        device="cpu",
        image_shape=None,
    )
    vm.voxel_pcd.add(torch.tensor([[0.0, 0.0, 0.5]]), features=None, rgb=torch.zeros(1, 3), obs_count=1)
    padded, explored = vm.get_2d_map()
    raw, physical_explored = vm.get_navigation_map()
    assert raw.sum() == 1
    assert padded.sum() > raw.sum()
    assert torch.all(~raw | padded)
    assert torch.equal(explored, physical_explored)
    assert vm.get_navigation_map()[0] is raw
    assert vm.get_2d_map()[0] is padded
    vm._seq += 1
    assert vm.get_navigation_map()[0] is not raw


def test_astar_and_footprint_use_same_physical_map(tmp_path):
    vm = SparseVoxelMap(
        grid_size=[40, 40],
        grid_resolution=0.1,
        pad_obstacles=2,
        smooth_kernel_size=0,
        obs_min_density=0.1,
        log=str(tmp_path),
        use_instance_memory=False,
        device="cpu",
        image_shape=None,
    )
    vm.voxel_pcd.add(torch.tensor([[0.0, 0.0, 0.5]]), features=None, rgb=torch.zeros(1, 3), obs_count=1)
    padded, explored = vm.get_2d_map()
    explored[:] = True
    footprint = Footprint(width=0.34, length=0.33, length_offset=-0.1)
    space = SparseVoxelMapNavigationSpace(vm, obstacle_map_mode="physical", footprint=footprint)
    legacy = SparseVoxelMapNavigationSpace(vm)
    physical_plan = AStar(space, min_clearance_m=0.22)
    legacy_plan = AStar(legacy, min_clearance_m=0.22)
    assert space._footprint is footprint
    assert np.count_nonzero(physical_plan._navigable) > np.count_nonzero(legacy_plan._navigable)
    # Removing double inflation must not allow occupying a measured obstacle.
    assert not space.is_valid(np.array([0.0, 0.0, 0.0]))
    assert not space.is_valid(np.array([1.9, 1.9, 0.0]))  # footprint outside grid
    assert space.get_navigation_map()[0].sum() < padded.sum()
