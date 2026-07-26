# Copyright (c) Chris Paxton 2026

"""Tests for frontier pick visualization panels."""

from __future__ import annotations

import numpy as np

from emet.visualization.frontier_pick_viz import (
    COLOR_EXPLORED,
    COLOR_FRONTIER,
    COLOR_OBSTACLE,
    COLOR_STAR,
    FrontierPickStep,
    frontier_mask_from_explored,
    make_long_motion_demo_steps,
    render_frontier_pick_rgb,
    write_frontier_pick_steps,
)


def test_frontier_mask_touches_explored_edge():
    exp = np.zeros((10, 10), dtype=bool)
    obs = np.zeros((10, 10), dtype=bool)
    exp[3:6, 3:6] = True
    fr = frontier_mask_from_explored(exp, obs)
    assert fr.any()
    assert not np.any(fr & exp)
    # Cell just outside the explored square must be frontier.
    assert fr[2, 4] or fr[6, 4] or fr[4, 2] or fr[4, 6]


def test_render_panel_has_title_and_layers(tmp_path):
    obs = np.zeros((40, 40), dtype=bool)
    exp = np.zeros((40, 40), dtype=bool)
    obs[0, :] = True
    obs[-1, :] = True
    obs[:, 0] = True
    obs[:, -1] = True
    exp[10:20, 10:20] = True
    go = np.array([20.0, 20.0])
    rgb = render_frontier_pick_rgb(
        obs,
        exp,
        robot_xy=(0.0, 0.0),
        chosen_xy=(0.5, 0.0),
        grid_origin_xy=go,
        grid_resolution=0.1,
        title="iteration 0 — pick 0.5 m ahead",
        max_side=320,
    )
    assert rgb.ndim == 3 and rgb.shape[2] == 3
    # Title banner is dark strip on top.
    assert tuple(rgb[2, rgb.shape[1] // 2]) == (24, 24, 32) or rgb[2, rgb.shape[1] // 2, 0] < 40
    # Map body contains the expected palette colors.
    flat = rgb.reshape(-1, 3)
    assert (flat == COLOR_EXPLORED).all(axis=1).any()
    assert (flat == COLOR_OBSTACLE).all(axis=1).any()
    assert (flat == COLOR_FRONTIER).all(axis=1).any()
    assert (flat == COLOR_STAR).all(axis=1).any()

    paths = write_frontier_pick_steps(
        [
            FrontierPickStep(
                iteration=0,
                obstacles=obs,
                explored=exp,
                robot_xy=(0.0, 0.0),
                chosen_xy=(0.5, 0.0),
                title="iteration 0",
            )
        ],
        tmp_path,
        grid_origin_xy=go,
        grid_resolution=0.1,
    )
    assert paths[0].is_file()
    assert paths[0].stat().st_size > 100


def test_numbered_waypoints_accumulate_on_panel():
    """Prior picks stay labeled 1..N so the path of goals is readable."""
    obs = np.zeros((50, 50), dtype=bool)
    exp = np.zeros((50, 50), dtype=bool)
    obs[0, :] = True
    obs[-1, :] = True
    obs[:, 0] = True
    obs[:, -1] = True
    exp[10:30, 10:30] = True
    go = np.array([25.0, 25.0])
    waypoints = [(-1.0, -1.0), (0.0, 0.0), (1.0, 0.5)]
    rgb = render_frontier_pick_rgb(
        obs,
        exp,
        robot_xy=(-1.2, -1.2),
        chosen_xy=waypoints[-1],
        waypoints=waypoints,
        grid_origin_xy=go,
        grid_resolution=0.1,
        title="iteration 2 — 3 waypoints",
        max_side=400,
    )
    assert rgb.ndim == 3
    flat = rgb.reshape(-1, 3)
    # Star still paints solid magenta; labels may anti-alias off pure white.
    assert (flat == COLOR_STAR).all(axis=1).any()
    assert (flat.max(axis=1) >= 200).any()
    assert len(waypoints) == 3


def test_long_motion_demo_picks_are_not_underfoot():
    steps, go = make_long_motion_demo_steps(n_iters=3)
    assert len(steps) == 3
    for step in steps:
        assert step.robot_xy is not None and step.chosen_xy is not None
        dist = float(
            np.hypot(
                step.chosen_xy[0] - step.robot_xy[0],
                step.chosen_xy[1] - step.robot_xy[1],
            )
        )
        # Demo is about committing to longer legs, not 0.5 m creep.
        assert dist >= 1.5, dist
        assert step.title.startswith("iteration")
    assert go.shape == (2,)
