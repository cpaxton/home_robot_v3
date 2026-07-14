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
    eval_topdown_map_rgb,
    format_navigation_report,
    overlay_trajectory_on_map_rgb,
    render_topdown_map_rgb,
    share_topdown_map_rgb,
    snapshot_eval_from_voxel_map,
    snapshot_from_voxel_map,
    world_xy_to_grid_ij,
    _dedupe_trajectory_xyt,
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


def test_build_map_stats_nearest_explored_when_base_off_blob():
    """Marker far from explored blob → nearest_explored_m reports the gap (frame-mismatch signal)."""
    obs = np.zeros((32, 32), dtype=bool)
    exp = np.zeros((32, 32), dtype=bool)
    exp[2:5, 2:5] = True
    go = np.array([16.0, 16.0])
    # Base at world (0,0) → grid (16,16); explored around (2,2) — far away
    stats = build_map_stats(obs, exp, go, 0.1, (0.0, 0.0))
    assert stats["base_on_explored_cell"] is False
    assert stats["nearest_explored_m"] is not None
    assert stats["nearest_explored_m"] > 0.5
    assert "nearest explored" in " ".join(stats["summary_lines"]).lower()


def test_snapshot_base_on_explored_when_robot_xy_matches_visited():
    from emet.mapping.voxel.voxel_dynamem import SparseVoxelMap

    vm = SparseVoxelMap(
        resolution=0.1,
        semantic_memory_resolution=0.1,
        feature_dim=3,
        use_instance_memory=False,
        encoder=None,
        device="cpu",
        map_2d_device="cpu",
        add_local_radius_points=True,
        local_radius=0.5,
    )
    world_xy = (1.5, -2.0)
    camera_pose = torch.eye(4, dtype=torch.float32)
    camera_pose[0, 3] = float(world_xy[0]) + 0.4
    camera_pose[1, 3] = float(world_xy[1])
    camera_pose[2, 3] = 1.2
    base_pose = torch.tensor([world_xy[0], world_xy[1], 0.0], dtype=torch.float32)
    rgb = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    xyz = torch.tensor([[world_xy[0] + 0.5, world_xy[1], 0.1]], dtype=torch.float32)
    vm.add(camera_pose, rgb, xyz=xyz, xyz_frame="world", base_pose=base_pose)

    _img, stats, _ = snapshot_from_voxel_map(vm, world_xy)
    assert stats["base_on_explored_cell"] is True
    assert stats.get("nearest_explored_m", 0.0) == 0.0

    # Episode-relative gps at origin (wrong frame) should miss the visited disk
    _img2, stats_gps, _ = snapshot_from_voxel_map(vm, (0.0, 0.0))
    assert stats_gps["base_on_explored_cell"] is False
    assert stats_gps["nearest_explored_m"] is not None
    assert stats_gps["nearest_explored_m"] > 0.5


def test_robot_base_xy_prefers_controller_world_frame():
    from emet.agent.tools import _robot_base_xy
    from unittest.mock import MagicMock

    robot = MagicMock()
    robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    agent = MagicMock()
    agent.world_base_xy.return_value = (-2.0, -0.5)
    executor = MagicMock()
    executor.agent = agent
    assert _robot_base_xy(robot, executor) == (-2.0, -0.5)
    # Without executor, falls back to raw gps
    assert _robot_base_xy(robot, None) == (0.0, 0.0)


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


def test_eval_topdown_map_masks_unexplored_margin_white():
    obs = np.zeros((128, 128), dtype=bool)
    exp = np.zeros((128, 128), dtype=bool)
    exp[50:60, 50:60] = True
    obs[55, 55] = True
    go = np.array([0.0, 0.0])
    share = share_topdown_map_rgb(obs, exp, go, 0.1, None, max_side=640)
    eval_map = eval_topdown_map_rgb(obs, exp, go, 0.1, None, max_side=640, min_map_side=0, margin_cells=16)
    assert eval_map.shape[0] >= 10
    white = np.all(eval_map == np.uint8([248, 248, 248]), axis=-1)
    green = np.all(eval_map == np.uint8([50, 160, 80]), axis=-1)
    red = np.all(eval_map == np.uint8([200, 55, 55]), axis=-1)
    assert int(green.sum()) > 0
    assert int(red.sum()) > 0
    assert int(white.sum()) > 0
    dark = np.all(eval_map < np.uint8([40, 40, 40]), axis=-1)
    assert int(dark.sum()) == 0


def test_snapshot_eval_from_voxel_map_uses_eval_style():
    class FakeVM:
        grid_origin = np.array([0.0, 0.0, 0.0])
        grid_resolution = 0.1

        def get_2d_map(self):
            obs = np.zeros((64, 64), dtype=bool)
            exp = np.zeros((64, 64), dtype=bool)
            exp[20:40, 20:40] = True
            obs[30, 30] = True
            return obs, exp

    img, stats = snapshot_eval_from_voxel_map(FakeVM(), (0.0, 0.0))
    assert img is not None
    assert stats["map_nonempty"]
    assert np.any(np.all(img == np.uint8([50, 160, 80]), axis=-1))
    assert not np.any(np.all(img < np.uint8([40, 40, 40]), axis=-1))


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


def test_dedupe_trajectory_drops_spin_in_place():
    raw = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)]
    deduped = _dedupe_trajectory_xyt(raw)
    assert len(deduped) == 2
    assert deduped[0][:2] == (0.0, 0.0)
    assert deduped[-1][:2] == (1.0, 0.0)


def test_eval_topdown_map_draws_trajectory_path():
    obs = np.zeros((64, 64), dtype=bool)
    exp = np.zeros((64, 64), dtype=bool)
    exp[10:50, 10:50] = True
    go = np.array([0.0, 0.0])
    traj = [(0.5, 0.5, 0.0), (1.5, 0.5, 1.57), (2.5, 1.5, 3.14)]
    before = eval_topdown_map_rgb(obs, exp, go, 0.1, (2.5, 1.5), max_side=640, min_map_side=0, trajectory_xyt=traj)
    plain = eval_topdown_map_rgb(obs, exp, go, 0.1, (2.5, 1.5), max_side=640, min_map_side=0)
    assert not np.array_equal(before, plain)
    blue = np.all(before == np.uint8([30, 90, 230]), axis=-1)
    assert int(blue.sum()) > 0
