# Copyright (c) Chris Paxton 2026

"""Unit tests for place approach sampling + frontier coverage completeness."""

from __future__ import annotations

import numpy as np

from emet.memory.graph_eqa.place_approaches import (
    PlaceFootprint,
    count_frontier_in_footprint,
    coverage_from_frontier_count,
    footprint_from_node,
    sample_annulus_approach_xy,
)


def test_coverage_from_frontier_count():
    assert coverage_from_frontier_count(None).status == "unknown"
    assert coverage_from_frontier_count(3).status == "open"
    assert coverage_from_frontier_count(3).local_frontier_cells == 3
    assert coverage_from_frontier_count(0).complete is True
    assert coverage_from_frontier_count(0).status == "closed"


def test_footprint_from_bounds_3d():
    node = type(
        "N",
        (),
        {
            "xyz": np.array([-16.5, -1.1, 0.7]),
            "bounds_3d": {
                "min": [-17.0, -1.5, 0.0],
                "max": [-16.0, -0.7, 1.0],
            },
            "extent_half": None,
        },
    )()
    fp = footprint_from_node(node)
    assert fp is not None
    assert abs(fp.cx - (-16.5)) < 1e-6
    assert abs(fp.half_x - 0.5) < 1e-6


def test_count_frontier_in_dilated_footprint():
    # 10x10 grid; frontier cell at (5,5). Footprint centered at world (0.5, 0.5).
    frontier = np.zeros((10, 10), dtype=bool)
    frontier[5, 5] = True
    fp = PlaceFootprint(cx=0.5, cy=0.5, half_x=0.2, half_y=0.2)

    def xy_to_ij(x: float, y: float):
        return (int(round(x * 10)), int(round(y * 10)))

    n = count_frontier_in_footprint(fp, frontier, xy_to_ij=xy_to_ij, dilate_m=0.3, resolution_m=0.1)
    assert n >= 1
    closed = count_frontier_in_footprint(
        PlaceFootprint(cx=9.0, cy=9.0, half_x=0.1, half_y=0.1),
        frontier,
        xy_to_ij=xy_to_ij,
        dilate_m=0.1,
        resolution_m=0.1,
    )
    assert closed == 0


def test_sample_annulus_avoids_obstacles_and_prior_xy():
    obstacles = np.zeros((21, 21), dtype=bool)
    obstacles[10, 10] = True  # object center cell
    # Block the entire right half so samples must land on the left.
    obstacles[:, 12:] = True

    def xy_to_ij(x: float, y: float):
        return (int(round(y + 10)), int(round(x + 10)))

    def ij_to_xy(i: int, j: int):
        return (float(j - 10), float(i - 10))

    xy = sample_annulus_approach_xy(
        anchor_xy=(0.0, 0.0),
        robot_xy=(-2.0, 0.0),
        obstacles=obstacles,
        reachable=None,
        frontier=None,
        footprint=PlaceFootprint(0.0, 0.0, 0.3, 0.3),
        xy_to_ij=xy_to_ij,
        ij_to_xy=ij_to_xy,
        avoid_xy=[(-1.0, 0.0)],
        radius_inner_m=0.5,
        radius_outer_m=1.5,
        approach_index=0,
        n_draws=80,
    )
    assert xy is not None
    assert xy[0] < 1.5  # not deep into blocked right half
    # Distinct second sample.
    xy2 = sample_annulus_approach_xy(
        anchor_xy=(0.0, 0.0),
        robot_xy=(-2.0, 0.0),
        obstacles=obstacles,
        reachable=None,
        frontier=None,
        footprint=PlaceFootprint(0.0, 0.0, 0.3, 0.3),
        xy_to_ij=xy_to_ij,
        ij_to_xy=ij_to_xy,
        avoid_xy=[xy],
        radius_inner_m=0.5,
        radius_outer_m=1.5,
        approach_index=1,
        n_draws=80,
    )
    assert xy2 is not None
    assert abs(xy2[0] - xy[0]) + abs(xy2[1] - xy[1]) > 0.2
