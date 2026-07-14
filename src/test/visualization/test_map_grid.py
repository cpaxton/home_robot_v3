# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from __future__ import annotations

import numpy as np

from emet.visualization.map_grid import prune_explored_islands


def test_prune_explored_islands_drops_remote_blob():
    exp = np.zeros((64, 64), dtype=bool)
    exp[20:40, 20:40] = True
    exp[5, 5] = True
    exp[6, 5] = True
    exp[5, 6] = True
    traj = [(2.5, 2.5, 0.0), (3.0, 3.0, 0.0)]
    go = np.array([0.0, 0.0])
    pruned = prune_explored_islands(
        exp,
        grid_origin_xy=go,
        grid_resolution=0.1,
        robot_xy=(3.0, 3.0),
        trajectory_xyt=traj,
        min_component_cells=4,
        anchor_radius_cells=6,
    )
    assert pruned[25, 25]
    assert not pruned[5, 5]


def test_trajectory_corridor_fills_sparse_export_gaps():
    from emet.visualization.map_grid import build_trajectory_corridor_mask, merge_trajectory_corridor_explored
    from emet.visualization.map_snapshot import eval_topdown_map_rgb

    obs = np.zeros((64, 64), dtype=bool)
    exp = np.zeros((64, 64), dtype=bool)
    exp[10, 10] = True
    exp[50, 50] = True
    go = np.array([0.0, 0.0])
    traj = [(1.0, 1.0, 0.0), (1.0, 4.0, 0.0), (4.0, 4.0, 0.0)]
    corridor = build_trajectory_corridor_mask((64, 64), go, 0.1, trajectory_xyt=traj, radius_cells=2)
    assert int(corridor.sum()) > 20
    merged = merge_trajectory_corridor_explored(exp, obs, go, 0.1, trajectory_xyt=traj)
    assert int(merged.sum()) > int(exp.sum())
    without = eval_topdown_map_rgb(
        obs, exp, go, 0.1, (4.0, 4.0), max_side=640, min_map_side=0, trajectory_xyt=traj, stamp_trajectory_corridor=False
    )
    with_stamp = eval_topdown_map_rgb(
        obs, exp, go, 0.1, (4.0, 4.0), max_side=640, min_map_side=0, trajectory_xyt=traj, stamp_trajectory_corridor=True
    )
    white_without = np.all(without == np.uint8([248, 248, 248]), axis=-1).sum()
    white_with = np.all(with_stamp == np.uint8([248, 248, 248]), axis=-1).sum()
    assert white_with < white_without


def test_eval_topdown_overlay_includes_trajectory_blue_pixels():
    from emet.visualization.map_snapshot import eval_topdown_map_rgb, eval_topdown_overlay_rgb

    obs = np.zeros((64, 64), dtype=bool)
    exp = np.zeros((64, 64), dtype=bool)
    exp[10:50, 10:50] = True
    go = np.array([0.0, 0.0])
    traj = [(1.0, 1.0, 0.0), (2.0, 2.0, 0.5), (3.0, 3.0, 1.0)]
    gt = np.zeros((64, 64), dtype=bool)
    gt[8:52, 8:52] = True
    plain = eval_topdown_map_rgb(
        obs, exp, go, 0.1, (3.0, 3.0), max_side=640, trajectory_xyt=traj, filter_islands=True
    )
    overlay = eval_topdown_overlay_rgb(
        obs,
        exp,
        go,
        0.1,
        (3.0, 3.0),
        gt_navigable=gt,
        max_side=640,
        trajectory_xyt=traj,
        filter_islands=True,
    )
    assert plain.shape[0] >= 20
    assert overlay.shape == plain.shape
    blue = np.all(overlay == np.uint8([30, 90, 230]), axis=-1)
    assert int(blue.sum()) > 0
