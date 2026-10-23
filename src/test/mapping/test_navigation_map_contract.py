# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
import numpy as np
import pytest
import torch

from emet.mapping.voxel.voxel_dynamem import SparseVoxelMap
from emet.mapping.voxel.voxel_map_dynamem import SparseVoxelMapNavigationSpace
from emet.motion.algo.a_star import AStar
from emet.robots.footprint import Footprint


@pytest.mark.parametrize("remove_visited", [False, True])
def test_dynamem_preserves_enabled_negative_obstacle_evidence(tmp_path, remove_visited):
    vm = SparseVoxelMap(
        grid_size=[40, 40],
        grid_resolution=0.1,
        smooth_kernel_size=0,
        obs_min_density=0.1,
        use_negative_obstacles=True,
        neg_obs_height=-0.1,
        remove_visited_from_obstacles=remove_visited,
        log=str(tmp_path),
        use_instance_memory=False,
        device="cpu",
        image_shape=None,
    )
    vm.voxel_pcd.add(torch.tensor([[0.0, 0.0, -0.2]]), features=None, rgb=torch.zeros(1, 3), obs_count=1)
    vm._visited[:] = 1
    obstacles, _ = vm.get_navigation_map()
    assert obstacles.any(), "A measured drop must not be flattened into traversable floor"


@pytest.mark.parametrize("xy", [(10.0, 0.0), (-2.2, 0.0), (0.0, 10.0), (0.0, -2.2)])
@pytest.mark.parametrize("grid_size", [[40, 40], [40, 20]])
def test_out_of_grid_negative_points_never_crash_or_wrap(tmp_path, xy, grid_size):
    vm = SparseVoxelMap(
        grid_size=grid_size,
        grid_resolution=0.1,
        smooth_kernel_size=0,
        obs_min_density=0.1,
        use_negative_obstacles=True,
        neg_obs_height=-0.1,
        remove_visited_from_obstacles=True,
        log=str(tmp_path),
        use_instance_memory=False,
        device="cpu",
        image_shape=None,
    )
    vm.voxel_pcd.add(torch.tensor([[*xy, -0.2]]), features=None, rgb=torch.zeros(1, 3), obs_count=1)
    # Isolate the new negative-density scatter from legacy scatter3d's boundary
    # clamping. An out-of-grid return must not wrap onto another valid cell.
    vm._visited[:] = 1
    obstacles, _ = vm.get_navigation_map()
    assert not obstacles.any()


@pytest.mark.parametrize("hazard", ["drops", "raised", "both"])
@pytest.mark.parametrize("tensor_maps", [False, True])
def test_caller_maps_cannot_override_measured_terrain_hazard(hazard, tensor_maps):
    from types import SimpleNamespace

    obstacles = np.zeros((9, 9), dtype=bool)
    explored = np.ones_like(obstacles)
    layers = SimpleNamespace(drops=obstacles.copy(), raised=obstacles.copy())
    if hazard == "both":
        layers.drops[4, 4] = layers.raised[4, 4] = True
    else:
        getattr(layers, hazard)[4, 4] = True
    space = object.__new__(SparseVoxelMapNavigationSpace)
    space.obstacle_map_mode = "physical"
    space._footprint = Footprint(0.1, 0.1)
    space.voxel_map = SimpleNamespace(
        grid_resolution=0.1, grid=SimpleNamespace(grid_origin=np.array([4, 4])), _floor_evidence=layers
    )
    supplied_obstacles = torch.from_numpy(obstacles) if tensor_maps else obstacles
    supplied_explored = torch.from_numpy(explored) if tensor_maps else explored
    assert not space.is_valid(np.zeros(3), obstacles=supplied_obstacles, explored=supplied_explored)
    assert space.last_validity["reason"] == "occupied_footprint"
    expected = ["drop", "raised_surface"] if hazard == "both" else ["drop" if hazard == "drops" else "raised_surface"]
    assert space.last_validity["terrain_hazard"] == expected[0]
    assert space.last_validity["terrain_hazards"] == expected
    assert [4, 4] in space.last_validity["occupied_cells"]
    assert not obstacles.any()  # Do not mutate caller-owned map snapshots.


def test_terrain_map_shape_mismatch_rejects_without_indexing():
    from types import SimpleNamespace

    space = object.__new__(SparseVoxelMapNavigationSpace)
    space.obstacle_map_mode = "physical"
    space._footprint = Footprint(0.1, 0.1)
    space.voxel_map = SimpleNamespace(
        grid_resolution=0.1,
        grid=SimpleNamespace(grid_origin=np.array([4, 4])),
        _floor_evidence=SimpleNamespace(drops=np.zeros((8, 8)), raised=np.zeros((9, 9))),
    )
    assert not space.is_valid(np.zeros(3), obstacles=np.zeros((9, 9)), explored=np.ones((9, 9)))
    assert space.last_validity["reason"] == "terrain_map_shape_mismatch"


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
    vm._visited[:] = 1
    padded, explored = vm.get_2d_map()
    raw, physical_explored = vm.get_navigation_map()
    assert raw.sum() == 1
    assert padded.sum() > raw.sum()
    assert torch.all(~raw | padded)
    assert torch.equal(explored, physical_explored)
    assert vm.get_navigation_map()[0] is raw
    assert vm.get_2d_map()[0] is padded
    sensor_cells = vm.get_sensor_observed_cells()
    assert sensor_cells.sum() == 1
    assert explored.sum() > sensor_cells.sum()
    sensor_cells[:] = False
    assert vm.get_sensor_observed_cells().sum() == 1  # Snapshot cannot mutate the map.
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
