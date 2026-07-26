# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for Dynagraph frontier graph nodes and question-guided exploration."""

from __future__ import annotations

import numpy as np

from emet.memory.graph_eqa import GraphEQAMemory
from emet.memory.graph_eqa.frontier_nodes import (
    cluster_frontier_mask,
    exploration_keywords_from_text,
    keyword_overlap_score,
    keyword_score_map,
    reachable_waypoint_for_cluster,
)


class _StubVoxelMap:
    def __init__(self, size: int = 12):
        self.shape = (size, size)
        self.grid_resolution = 0.05
        self.image_descriptions: list = []
        self.parameters = {
            "graph_eqa_frontier_nodes": {"keyword_score_weight": 2.0},
        }

    def get_outside_frontier(self, xyt, planner):
        mask = np.zeros(self.shape, dtype=bool)
        mask[2:6, 2:6] = True
        mask[8:10, 8:10] = True
        return mask

    def get_2d_map(self, return_history_id: bool = False, kernel: int = 5):
        explored = np.zeros(self.shape, dtype=bool)
        explored[:2, :] = True
        if return_history_id:
            history = np.ones(self.shape, dtype=np.float32)
            history[explored] = 5.0
            return None, explored, history
        return None, explored

    def grid_coords_to_xy(self, pt):
        return np.array([float(pt[0]) * 0.25, float(pt[1]) * 0.25], dtype=float)

    def xy_to_grid_coords(self, xy):
        return np.array([int(xy[0] / 0.25), int(xy[1] / 0.25)], dtype=int)


class _StubPlanner:
    def to_pt(self, xyt):
        return np.array([0, 0], dtype=int)


def test_cluster_frontier_mask_filters_small_components():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1, 1] = True
    mask[3:6, 3:6] = True
    clusters = cluster_frontier_mask(mask, min_cells=3)
    assert len(clusters) == 1
    assert clusters[0][2] >= 3


def test_reachable_waypoint_for_cluster_prefers_adjacent_not_centroid():
    """Ring/arc centroids land mid-floor; waypoints must sit on the reachable rim."""
    reachable = np.zeros((20, 20), dtype=bool)
    reachable[5:15, 5:15] = True
    frontier = np.zeros((20, 20), dtype=bool)
    # Full ring around the reachable block — geometric centroid is ~(9.5, 9.5) mid-floor.
    frontier[4, 5:15] = True
    frontier[15, 5:15] = True
    frontier[5:15, 4] = True
    frontier[5:15, 15] = True
    cells = np.argwhere(frontier)
    wp = reachable_waypoint_for_cluster(cells, reachable)
    assert wp is not None
    assert reachable[wp[0], wp[1]]
    # Must touch the frontier (8-adjacent to at least one frontier cell).
    fr, fc = wp
    assert any(
        frontier[fr + dr, fc + dc]
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr, dc) != (0, 0) and 0 <= fr + dr < 20 and 0 <= fc + dc < 20
    )
    # Not the mid-floor centroid.
    assert wp != (10, 10)


def test_cluster_frontier_mask_snaps_to_reachable_when_provided():
    reachable = np.zeros((16, 16), dtype=bool)
    reachable[4:12, 4:12] = True
    frontier = np.zeros((16, 16), dtype=bool)
    frontier[3, 4:12] = True  # top edge outside reachable
    clusters = cluster_frontier_mask(frontier, min_cells=3, reachable=reachable)
    assert len(clusters) == 1
    gi, gj = clusters[0][1]
    assert reachable[gi, gj]
    assert gi == 4  # just inside the reachable block
    # Without reachable, goal is the arc centroid (outside / mid-edge).
    raw = cluster_frontier_mask(frontier, min_cells=3, reachable=None)
    assert raw[0][1][0] == 3


def test_sync_frontier_nodes_creates_and_removes():
    mem = GraphEQAMemory(
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
        parameters={"graph_eqa_frontier_nodes": {"enabled": True, "max_nodes": 4, "min_cluster_cells": 2}},
    )
    vm = _StubVoxelMap()
    planner = _StubPlanner()
    n0 = mem.sync_frontier_nodes(vm, planner, [0, 0, 0])
    assert n0 >= 1
    assert any(n.is_frontier for n in mem.get_nodes())
    # IMAGE_DESCRIPTIONS now describes only attached images; pass the frontier obs ids.
    frontier_obs_ids = [int(n.obs_id) for n in mem.get_nodes() if n.is_frontier]
    desc = mem._get_image_descriptions_str(frontier_obs_ids)
    assert "unexplored" in desc.lower()

    explored = np.ones(vm.shape, dtype=bool)
    vm.get_2d_map = lambda return_history_id=False, kernel=5: (None, explored)
    n1 = mem.sync_frontier_nodes(vm, planner, [0, 0, 0])
    assert n1 == 0
    assert not any(n.is_frontier for n in mem.get_nodes())


def test_sync_frontier_nodes_snaps_xyz_to_reachable_adjacent():
    """With get_reachable_map, frontier node xyz must not be the mid-arc centroid."""

    class _ReachableStub(_StubVoxelMap):
        def get_outside_frontier(self, xyt, planner):
            mask = np.zeros(self.shape, dtype=bool)
            # Arc along top of a reachable block — centroid y would be mid-row of arc.
            mask[2, 2:8] = True
            return mask

        def get_2d_map(self, return_history_id: bool = False, kernel: int = 5):
            explored = np.zeros(self.shape, dtype=bool)
            return None, explored

        def get_reachable_map(self, xyt, planner, *, local_fallback_cells: int = 8):
            reachable = np.zeros(self.shape, dtype=bool)
            reachable[3:9, 2:8] = True
            return reachable

    mem = GraphEQAMemory(
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
        parameters={"graph_eqa_frontier_nodes": {"enabled": True, "max_nodes": 4, "min_cluster_cells": 2}},
    )
    vm = _ReachableStub()
    mem.sync_frontier_nodes(vm, _StubPlanner(), [0, 0, 0])
    frontiers = [n for n in mem.get_nodes() if n.is_frontier]
    assert frontiers
    # Goal cell snapped onto reachable row 3, not frontier-arc row 2.
    gi = int(round(float(frontiers[0].xyz[0]) / 0.25))
    gj = int(round(float(frontiers[0].xyz[1]) / 0.25))
    assert gi == 3
    assert 2 <= gj < 8
    reachable = vm.get_reachable_map(None, None)
    assert reachable[gi, gj]


def test_keyword_overlap_score():
    assert keyword_overlap_score(["table", "monitor"], ["table", "laptop"]) == 0.5
    assert keyword_overlap_score([], ["table"]) == 0.0


def test_keyword_score_map_biases_toward_matching_labels():
    vm = _StubVoxelMap()
    outside = vm.get_outside_frontier(None, None)
    _, explored = vm.get_2d_map()
    unexplored = outside & ~explored
    vm.image_descriptions = [
        (["bathtub", "sink"], np.array([8.0, 8.0])),
        (["table", "monitor"], np.array([4.0, 4.0])),
    ]
    keywords = exploration_keywords_from_text("Where is my laptop on the table?")
    scores = keyword_score_map(unexplored, vm.image_descriptions, keywords)
    peak = np.unravel_index(int(np.argmax(scores)), scores.shape)
    dist_table = np.hypot(peak[0] - 4, peak[1] - 4)
    dist_bath = np.hypot(peak[0] - 8, peak[1] - 8)
    assert dist_table < dist_bath


def test_siglip_activation_map_biases_toward_aligned_observation():
    """SigLIP activation sampling boosts frontier cells near the best-aligned observation,
    independent of caption keywords. No-ops when no encoder is present (clean baseline)."""
    from types import SimpleNamespace

    import torch

    from emet.mapping.voxel.voxel_map_dynamem import SparseVoxelMapNavigationSpace as NS

    pts = torch.tensor([[0.0, 0.0, 0.5], [2.0, 3.0, 0.5], [9.0, 9.0, 0.5]])
    align = torch.tensor([[0.05, 0.30, 0.04]])  # peak at the (2.0, 3.0) point
    vm = SimpleNamespace(
        encoder=object(),
        grid_resolution=0.1,
        grid_origin=np.array([0.0, 0.0, 0.0]),
        find_alignment_over_model=lambda _t: align,
        semantic_memory=SimpleNamespace(get_pointcloud=lambda: (pts, None, None, None)),
        parameters=None,
    )
    fake = SimpleNamespace(voxel_map=vm)
    mask = np.ones((120, 120), dtype=bool)
    out = NS._siglip_activation_map(fake, "basket", mask, radius_cells=10)
    assert out is not None
    assert float(out.max()) == 1.0
    peak = np.unravel_index(int(np.argmax(out)), out.shape)
    # Aligned point projects near grid cell (20, 30); peak should be within the spread radius.
    assert np.hypot(peak[0] - 20, peak[1] - 30) <= 10
    # Far frontier (near the (9,9)->(90,90) cell) stays unboosted.
    assert out[90, 90] == 0.0

    # Weak/floor-level alignment (no genuine match) -> inert, no spurious trajectory bias.
    vm.find_alignment_over_model = lambda _t: torch.tensor([[0.05, 0.12, 0.04]])
    assert NS._siglip_activation_map(fake, "basket", mask, radius_cells=10) is None

    # No encoder -> heuristic is inert (baseline GraphEQA).
    vm.encoder = None
    vm.find_alignment_over_model = lambda _t: align
    assert NS._siglip_activation_map(fake, "basket", mask, radius_cells=10) is None


def test_exploration_keywords_from_text_includes_heuristic_tokens():
    keys = exploration_keywords_from_text("Where is the red cup on the kitchen table?")
    assert any("cup" in k or "table" in k or "kitchen" in k for k in keys)


def test_query_answer_image_descriptions_tag_frontier():
    mem = GraphEQAMemory(
        eqa_client=lambda x: "reasoning: r\nanswer: a\nconfidence: false\naction: 1\nconfidence_reasoning: need more",
        image_description_client=lambda x: "",
        parameters={"graph_eqa_frontier_nodes": {"enabled": True, "min_cluster_cells": 2}},
    )
    vm = _StubVoxelMap()
    mem.sync_frontier_nodes(vm, _StubPlanner(), [0, 0, 0])
    frontier_ids = [int(o.obs_id) for o in mem.get_observations() if mem._obs_is_frontier(o.obs_id)]
    assert frontier_ids
    # Descriptions may still tag frontiers for debug / SCENE_GRAPH Action targets.
    s = mem._get_image_descriptions_str(frontier_ids)
    assert "unexplored" in s.lower()
    assert s.count("Image ") == len(frontier_ids)
    assert "Image 4." not in s
    # Answer-image selection must never attach placeholder frontiers.
    mem.extract_relevant_objects("explore the room")
    obs_ids = mem._select_relevant_obs_ids(max_images=3)
    assert not any(mem._obs_is_frontier(oid) for oid in obs_ids)

    obs = mem.get_observations()[0]
    pt = mem._target_point_from_image_id(obs.obs_id)
    assert pt is not None
    assert pt.shape == (3,)
