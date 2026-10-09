# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
import numpy as np

from emet.motion.placement import surface_placement_centers
from emet.motion.placement_geometry import HeldObject, PlacementScene
from emet.motion.placement_surfaces import free_surface_centers


def payload():
    return HeldObject.from_world_bounds([[-.05]*3, [.05]*3], ee_position=[0, 0, 0], ee_rotation=np.eye(3))


def test_finds_off_grid_gap_for_full_object():
    support = [[0, 0, 0], [1, 1, 0]]
    # A usable but off-grid gap at x=.34: none of the 5x5 grid's x coordinates fit.
    scene = PlacementScene([[[0, 0, .001], [.25, 1, .4]], [[.43, 0, .001], [1, 1, .4]]],
                           source="observed_voxels", workspace=[[-1]*3, [2]*3])
    obj = payload()
    grid = surface_placement_centers(support, payload=obj, ee_rotation=np.eye(3))
    assert all(scene.collides([point - .05, point + .05]) for point in grid)
    result = free_surface_centers(support, scene=scene, payload=obj, ee_rotation=np.eye(3))
    assert result.centers
    for point in result.centers:
        assert .305 < point[0] < .375
        assert not scene.collides([point - .055, point + .055 + [0, 0, .12]])
    assert result.blocker_bounds and not result.budget_exhausted


def test_blocks_vertical_corridor_even_with_clear_endpoints():
    scene = PlacementScene([[[0, 0, .15], [1, 1, .16]]], source="ground_truth")
    result = free_surface_centers([[0, 0, 0], [1, 1, 0]], scene=scene, payload=payload(), ee_rotation=np.eye(3))
    assert not result.centers
    assert len(result.blocker_bounds) == 1


def test_disjoint_support_patches_do_not_bridge_gap():
    scene = PlacementScene([], source="ground_truth")
    patches = [[[0, 0, 0], [.06, 1, 0]], [[.3, 0, 0], [.36, 1, 0]]]
    assert not free_surface_centers(patches, scene=scene, payload=payload(), ee_rotation=np.eye(3)).centers


def test_unknown_space_and_candidate_budget():
    scene = PlacementScene([], source="observed_voxels", workspace=[[-1]*3, [2]*3],
                           known_free=lambda bounds: bounds[1, 0] < .5)
    result = free_surface_centers([[0, 0, 0], [1, 1, 0]], scene=scene, payload=payload(),
                                  ee_rotation=np.eye(3), max_centers=1)
    assert len(result.centers) == 1
    assert result.centers[0][0] + .055 < .5


def test_candidate_truncation_is_reported_only_when_distinct_centers_are_omitted():
    scene = PlacementScene([], source="ground_truth")
    args = dict(scene=scene, payload=payload(), ee_rotation=np.eye(3))
    small = free_surface_centers([[0, 0, 0], [1, 1, 0]], max_centers=1, **args)
    exact = free_surface_centers([[0, 0, 0], [1, 1, 0]], max_centers=5, **args)
    assert len(small.centers) == 1 and small.budget_exhausted
    assert len(exact.centers) == 5 and not exact.budget_exhausted
