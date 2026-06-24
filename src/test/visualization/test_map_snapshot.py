# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

from __future__ import annotations

import numpy as np
import torch

from emet.agent.tools import get_tools
from emet.mapping.voxel.voxel_dynamem import _apply_map_boundary_2d, _map_boundary_config
from emet.visualization.map_snapshot import (
    build_map_stats,
    discord_share_map_rgb,
    format_navigation_report,
    render_topdown_map_rgb,
    share_topdown_map_rgb,
    snapshot_from_voxel_map,
    world_xy_to_grid_ij,
)


def test_world_xy_to_grid_ij_respects_full_shape_when_cropping():
    """Crop offset math: grid index from full map, then subtract crop origin."""
    go = np.array([100.0, 100.0])
    res = 0.05
    full_shape = (200, 200)
    i0, j0 = 40, 50
    xy = np.array([0.0, 0.0], dtype=np.float64)
    ri, rj = world_xy_to_grid_ij(xy, go, res, full_shape)
    assert (ri, rj) == (100, 100)
    assert ri - i0 == 60
    assert rj - j0 == 50


def test_world_xy_to_grid_ij_clamps():
    go = np.array([5.0, 5.0])
    i, j = world_xy_to_grid_ij((0.0, 0.0), go, 0.1, (10, 10))
    assert 0 <= i < 10
    assert 0 <= j < 10


def test_render_topdown_nonzero_with_explored():
    obs = np.zeros((16, 16), dtype=bool)
    exp = np.zeros((16, 16), dtype=bool)
    exp[4:12, 4:12] = True
    obs[8, 8] = True
    go = np.array([8.0, 8.0])
    rgb = render_topdown_map_rgb(obs, exp, go, 0.1, robot_xy=(0.0, 0.0), max_side=256)
    assert rgb.dtype == np.uint8
    assert rgb.shape[2] == 3
    assert rgb.max() > 0


def test_build_map_stats_summary_lines():
    obs = np.zeros((8, 8), dtype=bool)
    exp = np.zeros((8, 8), dtype=bool)
    exp[2:6, 2:6] = True
    go = np.array([4.0, 4.0])
    stats = build_map_stats(obs, exp, go, 0.1, (0.0, 0.0))
    assert stats["free_explored_cells"] > 0
    assert "2D map shape" in " ".join(stats["summary_lines"])


def test_format_navigation_report_explore_flag():
    stats = {"summary_lines": ["line a.", "line b."]}
    s = format_navigation_report(stats, explore_ok=False)
    assert "Last explore command" in s
    assert "failure" in s


def test_snapshot_from_voxel_map_fake_crops_to_explored():
    class FakeVM:
        grid_origin = np.array([0.0, 0.0, 0.0])
        grid_resolution = 0.1

        def get_2d_map(self):
            obs = np.zeros((128, 128), dtype=bool)
            exp = np.zeros((128, 128), dtype=bool)
            exp[50:60, 50:60] = True
            return obs, exp

    vm = FakeVM()
    obs, exp = vm.get_2d_map()
    go = np.array(vm.grid_origin[:2])
    img, stats, img_share = snapshot_from_voxel_map(vm, (0.0, 0.0))
    full = render_topdown_map_rgb(obs, exp, go, vm.grid_resolution, None, max_side=None)
    assert img is not None
    assert img_share is img
    assert stats["map_nonempty"]
    assert img.shape[0] < full.shape[0] or img.shape[1] < full.shape[1]


def test_map_boundary_config_defaults():
    assert _map_boundary_config(None) == (0, 0)
    assert _map_boundary_config({}) == (0, 0)
    assert _map_boundary_config({"map_boundary": {"obstacle_barrier_cells": 30, "history_penalty_cells": 35}}) == (
        30,
        35,
    )


def test_apply_map_boundary_2d_off_by_default():
    obs = torch.zeros(40, 40, dtype=torch.bool)
    _apply_map_boundary_2d(obs, None, None)
    assert not obs.any()


def test_apply_map_boundary_2d_marks_edge_when_enabled():
    obs = torch.zeros(40, 40, dtype=torch.bool)
    _apply_map_boundary_2d(obs, None, {"map_boundary": {"obstacle_barrier_cells": 5}})
    assert obs[0, 0].item()
    assert not obs[20, 20].item()


def test_share_topdown_map_crops_large_grid_to_explored_patch():
    obs = np.zeros((128, 128), dtype=bool)
    exp = np.zeros((128, 128), dtype=bool)
    exp[50:60, 50:60] = True
    go = np.array([0.0, 0.0])
    full = render_topdown_map_rgb(obs, exp, go, 0.1, None, max_side=None)
    cropped = share_topdown_map_rgb(obs, exp, go, 0.1, None, max_side=640)
    assert cropped.shape[0] < full.shape[0] or cropped.shape[1] < full.shape[1]
    assert np.array_equal(cropped, discord_share_map_rgb(obs, exp, go, 0.1, None, max_side=640))


def test_explore_tool_returns_diagnostic_text():
    class FakeVM:
        grid_origin = np.array([2.0, 2.0, 0.0])
        grid_resolution = 0.1

        def get_2d_map(self):
            o = np.zeros((5, 5), dtype=bool)
            e = np.zeros((5, 5), dtype=bool)
            e[1:4, 1:4] = True
            return o, e

    class FakeAgent:
        def get_voxel_map(self):
            return FakeVM()

    class FakeExec:
        agent = FakeAgent()

        def __call__(self, cmds):
            assert cmds == [("explore", "")]
            return False

    ctx: dict = {"executor": FakeExec(), "robot": None}
    by_name = {t.name: t for t in get_tools(ctx)}
    ex = by_name["explore"]
    assert ex.to_executor({}) == []
    out = ex.func()
    assert "Explore failed" in out
    assert "2D map shape" in out
