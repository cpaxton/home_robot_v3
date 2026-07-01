# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import numpy as np

from emet.mapping.debug_navgrid_ascii import (
    NavGridSnapshot,
    NavOverlay,
    render_navgrid_ascii,
    snapshot_from_numpy_2d,
)


def test_render_navgrid_ascii_obstacle_and_explored():
    obs = np.zeros((40, 40), dtype=bool)
    exp = np.zeros((40, 40), dtype=bool)
    obs[10:15, 10:15] = True
    exp[5:25, 5:25] = True
    snap = snapshot_from_numpy_2d(obs, exp, grid_resolution_m=0.1, grid_origin_xy=(20.0, 20.0))
    text = render_navgrid_ascii(snap, max_side=40)
    assert "navgrid_key:" in text
    assert "#" in text
    assert "." in text
    assert text.count("#") >= 2
    assert text.count(".") >= 2


def test_render_navgrid_ascii_semantic_glyph_and_legend():
    obs = np.zeros((30, 30), dtype=bool)
    exp = np.ones((30, 30), dtype=bool)
    snap = NavGridSnapshot(
        obstacles=obs,
        explored=exp,
        grid_resolution_m=0.1,
        grid_origin_xy=(15.0, 15.0),
    )
    overlay = NavOverlay(
        kind="graph_point",
        key="node:3",
        xy_min=(0.0, 0.0),
        xy_max=(0.5, 0.5),
        confidence=4.0,
        labels=("chair",),
        caption="wooden chair near table",
    )
    text = render_navgrid_ascii(snap, overlays=(overlay,), robot_xy=(0.2, 0.2), max_side=30)
    assert "Legend:" in text
    assert "node:3" in text
    assert "chair" in text
    assert "@" in text or "0" in text


def test_render_navgrid_ascii_empty_map_message():
    obs = np.zeros((10, 10), dtype=bool)
    exp = np.zeros((10, 10), dtype=bool)
    snap = snapshot_from_numpy_2d(obs, exp, grid_resolution_m=0.1, grid_origin_xy=(5.0, 5.0))
    text = render_navgrid_ascii(snap)
    assert "no explored cells" in text


def test_render_navgrid_ascii_explored_crop_not_full_grid():
    """Small explored patch in a large grid should crop tightly (Discord-style)."""
    obs = np.zeros((512, 512), dtype=bool)
    exp = np.zeros((512, 512), dtype=bool)
    exp[240:280, 240:280] = True
    obs[250:260, 250:260] = True
    snap = snapshot_from_numpy_2d(obs, exp, grid_resolution_m=0.1, grid_origin_xy=(256.0, 256.0))
    text = render_navgrid_ascii(snap, max_side=320)
    assert "explored crop grid[" in text
    assert "0:512,0:512" not in text
    assert "stride=1" in text
    assert "#" in text
    assert "." in text
