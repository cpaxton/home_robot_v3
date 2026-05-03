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

from emet.agent.tools import get_tools
from emet.visualization.map_snapshot import (
    build_map_stats,
    discord_share_map_rgb,
    format_navigation_report,
    render_topdown_map_rgb,
    snapshot_from_voxel_map,
    world_xy_to_grid_ij,
)


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


def test_snapshot_from_voxel_map_fake():
    class FakeVM:
        grid_origin = np.array([3.0, 3.0, 0.0])
        grid_resolution = 0.05

        def get_2d_map(self):
            obs = np.zeros((6, 6), dtype=bool)
            exp = np.zeros((6, 6), dtype=bool)
            exp[1:5, 1:5] = True
            return obs, exp

    img, stats, img_discord = snapshot_from_voxel_map(FakeVM(), (0.0, 0.0))
    assert img is not None
    assert img_discord is not None
    assert stats["map_nonempty"]
    assert img_discord.shape[0] <= img.shape[0]
    assert img_discord.shape[1] <= img.shape[1]


def test_discord_share_map_crops_large_grid_to_explored_patch():
    obs = np.zeros((128, 128), dtype=bool)
    exp = np.zeros((128, 128), dtype=bool)
    exp[50:60, 50:60] = True
    go = np.array([0.0, 0.0])
    full = render_topdown_map_rgb(obs, exp, go, 0.1, None, max_side=None)
    disc = discord_share_map_rgb(obs, exp, go, 0.1, None, max_side=640)
    assert disc.shape[0] < full.shape[0] or disc.shape[1] < full.shape[1]


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
