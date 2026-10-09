# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import numpy as np

from emet.mapping.navgrid_compare import (
    WorldMapRaster,
    binary_iou,
    compare_world_rasters,
    compare_world_rasters_in_shared_view,
    format_similarity_table,
    render_world_raster_ascii,
)


def test_binary_iou_identical_and_disjoint():
    a = np.zeros((10, 10), dtype=bool)
    a[2:5, 2:5] = True
    assert binary_iou(a, a) == 1.0
    b = np.zeros((10, 10), dtype=bool)
    b[7:9, 7:9] = True
    assert binary_iou(a, b) == 0.0


def test_compare_world_rasters_partial_overlap():
    exp = np.zeros((20, 20), dtype=bool)
    exp[5:15, 5:15] = True
    obs = np.zeros((20, 20), dtype=bool)
    obs[8:10, 8:10] = True
    meta = {"resolution_m": 0.1, "origin_xy": (0.0, 0.0), "clip_rect": (0.0, 2.0, 0.0, 2.0)}
    a = WorldMapRaster(explored=exp, obstacles=obs, **meta)
    b = WorldMapRaster(explored=exp.copy(), obstacles=obs.copy(), **meta)
    b.explored[12:18, 12:18] = True
    sim = compare_world_rasters(a, b)
    assert 0.4 < sim.explored_iou < 1.0
    assert sim.obstacle_iou == 1.0


def test_compare_world_rasters_in_shared_view():
    exp = np.zeros((20, 20), dtype=bool)
    exp[5:15, 5:15] = True
    obs = np.zeros((20, 20), dtype=bool)
    obs[8:10, 8:10] = True
    meta = {"resolution_m": 0.1, "origin_xy": (0.0, 0.0), "clip_rect": (0.0, 2.0, 0.0, 2.0)}
    a = WorldMapRaster(explored=exp, obstacles=obs, **meta)
    b = WorldMapRaster(explored=exp.copy(), obstacles=obs.copy(), **meta)
    b.explored[12:18, 12:18] = True
    b.obstacles[14:16, 14:16] = True
    full = compare_world_rasters(a, b)
    shared = compare_world_rasters_in_shared_view(a, b)
    assert shared.explored_iou >= full.explored_iou
    assert 0.0 <= shared.obstacle_iou <= 1.0


def test_format_similarity_table_and_render():
    exp = np.zeros((8, 8), dtype=bool)
    exp[2:6, 2:6] = True
    obs = np.zeros((8, 8), dtype=bool)
    obs[3:5, 3:5] = True
    meta = {"resolution_m": 0.1, "origin_xy": (0.0, 0.0), "clip_rect": (0.0, 0.8, 0.0, 0.8)}
    stretch = WorldMapRaster(explored=exp, obstacles=obs, **meta)
    rby1 = WorldMapRaster(explored=exp.copy(), obstacles=obs.copy(), **meta)
    table = format_similarity_table(["stretch", "rby1"], {"stretch": stretch, "rby1": rby1}, reference="stretch")
    assert "stretch" in table and "rby1" in table
    assert "1.000" in table
    text = render_world_raster_ascii(stretch, max_side=16)
    assert "#" in text and "." in text


def test_occupancy_world_coordinates_use_grid_resolution_not_voxel_size():
    """Different 3D/2D resolutions must not shrink mapped obstacle locations."""
    import torch

    from emet.mapping.voxel.voxel import SparseVoxelMap

    vm = SparseVoxelMap(
        resolution=0.03,
        grid_resolution=0.05,
        grid_size=(100, 100),
        obs_min_density=0,
        smooth_kernel_size=0,
        use_instance_memory=False,
        add_local_radius_points=False,
    )
    point = torch.tensor([[0.52, -0.98, 0.5]])
    vm.voxel_pcd.add(point, features=None, rgb=torch.ones((1, 3)))
    obstacles, _ = vm.get_2d_map()
    coords = vm.grid.xy_to_grid_coords(point[:, :2]).long()[0]
    assert obstacles[coords[0], coords[1]], "World-to-grid conversion misses the actual obstacle cell"
    recovered = vm.grid.grid_coords_to_xy(torch.nonzero(obstacles).float())
    assert len(recovered) == 1
    np.testing.assert_allclose(recovered.numpy()[0], point.numpy()[0, :2], atol=0.05)
    assert vm.voxel_resolution == 0.03
    assert vm.grid.resolution == vm.grid_resolution == 0.05
