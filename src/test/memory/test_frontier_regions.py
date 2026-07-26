# Copyright (c) Chris Paxton 2026

"""Region-level frontier ranking: area gain per unit travel, plus escape mode.

Regression target is HM-EQA holdout q104/q105, where nearest-first selection kept the
robot inside a 4.6 m box for 16 m of walking while whole rooms stayed unexplored.
"""

from __future__ import annotations

import numpy as np
import pytest

from emet.memory.graph_eqa.frontier_nodes import cluster_frontier_mask
from emet.memory.graph_eqa.frontier_regions import (
    FrontierRegion,
    frontier_region_utility,
    rank_frontier_regions,
    region_from_node,
)


class _Node:
    """Stand-in for a frontier GraphNode."""

    def __init__(self, xyz, cell_count=0, keyword_score=0.0, nav_failures=0, obs_id=0):
        self.xyz = np.asarray(xyz, dtype=float)
        self.frontier_cell_count = cell_count
        self.frontier_keyword_score = keyword_score
        self.nav_failures = nav_failures
        self.obs_id = obs_id
        self.last_seen = 0
        self.description = f"frontier:{obs_id}"


def test_large_distant_region_outranks_sliver_underfoot():
    """The q104 failure: a 2-cell frontier 0.5 m away beat the room 8 m away."""
    sliver = FrontierRegion(region_id="near", xy=(0.5, 0.0), cell_count=4)
    room = FrontierRegion(region_id="far", xy=(8.0, 0.0), cell_count=3000)
    ranked = rank_frontier_regions([sliver, room], (0.0, 0.0))
    assert [r.region_id for r in ranked] == ["far", "near"]


def test_nearest_wins_when_area_is_equal():
    near = FrontierRegion(region_id="near", xy=(1.0, 0.0), cell_count=500)
    far = FrontierRegion(region_id="far", xy=(9.0, 0.0), cell_count=500)
    ranked = rank_frontier_regions([near, far], (0.0, 0.0))
    assert [r.region_id for r in ranked] == ["near", "far"]


def test_missing_cluster_metadata_falls_back_to_proximity():
    """MuJoCo/legacy nodes carry no cell counts; ordering must stay nearest-first."""
    a = FrontierRegion(region_id="a", xy=(3.0, 0.0))
    b = FrontierRegion(region_id="b", xy=(1.0, 0.0))
    ranked = rank_frontier_regions([a, b], (0.0, 0.0))
    assert [r.region_id for r in ranked] == ["b", "a"]


def test_keyword_affinity_breaks_ties_between_equal_regions():
    plain = FrontierRegion(region_id="plain", xy=(4.0, 0.0), cell_count=400)
    relevant = FrontierRegion(region_id="relevant", xy=(4.0, 0.1), cell_count=400, keyword_score=1.0)
    ranked = rank_frontier_regions([plain, relevant], (0.0, 0.0))
    assert ranked[0].region_id == "relevant"


def test_nav_failures_decay_utility():
    healthy = FrontierRegion(region_id="ok", xy=(5.0, 0.0), cell_count=800)
    failing = FrontierRegion(region_id="bad", xy=(5.0, 0.0), cell_count=800, nav_failures=3)
    u_ok = frontier_region_utility(healthy, (0.0, 0.0))
    u_bad = frontier_region_utility(failing, (0.0, 0.0))
    assert u_bad < u_ok
    assert u_bad == pytest.approx(u_ok * 0.5**3)


def test_recent_goal_demotes_region():
    region = FrontierRegion(region_id="r", xy=(2.0, 0.0), cell_count=600)
    plain = frontier_region_utility(region, (0.0, 0.0))
    revisit = frontier_region_utility(region, (0.0, 0.0), recent=[(2.1, 0.0)])
    assert revisit < plain


def test_escape_floor_demotes_regions_inside_radius():
    """After repeated 'not visible' views, the nearby cluster must lose to the far one."""
    near = FrontierRegion(region_id="near", xy=(1.0, 0.0), cell_count=900)
    far = FrontierRegion(region_id="far", xy=(6.0, 0.0), cell_count=300)
    assert rank_frontier_regions([near, far], (0.0, 0.0))[0].region_id == "near"
    escaped = rank_frontier_regions([near, far], (0.0, 0.0), min_travel_m=3.0)
    assert escaped[0].region_id == "far"


def test_region_from_node_reads_cluster_metadata():
    node = _Node((2.0, 3.0, 0.0), cell_count=120, keyword_score=0.5, nav_failures=1, obs_id=7)
    region = region_from_node(node)
    assert region.xy == (2.0, 3.0)
    assert region.cell_count == 120
    assert region.keyword_score == 0.5
    assert region.nav_failures == 1
    assert region.area_m2(0.1) == pytest.approx(1.2)


def test_ranking_on_clustered_synthetic_grid():
    """End-to-end on a grid: a big far room should be preferred over a near sliver."""
    grid = np.zeros((80, 80), dtype=bool)
    grid[2:4, 2:4] = True  # tiny cluster next to the robot
    grid[50:75, 50:75] = True  # large unexplored room across the map
    clusters = cluster_frontier_mask(grid, min_cells=3)
    assert len(clusters) == 2

    resolution = 0.1
    regions = [
        FrontierRegion(
            region_id=cid,
            xy=(float(ij[0]) * resolution, float(ij[1]) * resolution),
            cell_count=count,
        )
        for cid, ij, count in clusters
    ]
    ranked = rank_frontier_regions(regions, (0.3, 0.3), grid_resolution_m=resolution)
    assert ranked[0].cell_count == max(r.cell_count for r in regions)


def test_habitat_sort_key_prefers_large_region():
    from emet.controller.habitat_nav import _frontier_explore_sort_key

    near = _Node((1.0, 0.0, 1.0), cell_count=9, obs_id=1)
    far = _Node((7.0, 0.0, 1.0), cell_count=2500, obs_id=2)
    ordered = sorted(
        [near, far],
        key=lambda n: _frontier_explore_sort_key(n, (0.0, 0.0), grid_resolution_m=0.1),
    )
    assert ordered[0].obs_id == 2


def test_habitat_sort_key_without_metadata_is_nearest_first():
    from emet.controller.habitat_nav import _frontier_explore_sort_key

    a = _Node((4.0, 0.0, 1.0), obs_id=1)
    b = _Node((1.0, 0.0, 1.0), obs_id=2)
    ordered = sorted([a, b], key=lambda n: _frontier_explore_sort_key(n, (0.0, 0.0)))
    assert ordered[0].obs_id == 2
