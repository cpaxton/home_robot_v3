# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import queue

import numpy as np

from emet.controller.nav_confirm import (
    confirm_navigation_plan,
    finite_traj_xyt,
    parse_nav_confirm_reply,
    render_nav_plan_map_rgb,
    wait_for_nav_confirm,
)


def test_parse_nav_confirm_reply_yes_no():
    assert parse_nav_confirm_reply("y") is True
    assert parse_nav_confirm_reply("YES") is True
    assert parse_nav_confirm_reply("[discord] yes") is True
    assert parse_nav_confirm_reply("@virgil go") is True
    assert parse_nav_confirm_reply("n") is False
    assert parse_nav_confirm_reply("cancel") is False
    assert parse_nav_confirm_reply("[discord] nope") is False
    assert parse_nav_confirm_reply("maybe") is None
    assert parse_nav_confirm_reply("") is None


def test_finite_traj_xyt_skips_nan():
    traj = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [np.nan, np.nan, np.nan], [2.0, 1.0, 1.5]]
    pts = finite_traj_xyt(traj)
    assert len(pts) == 3
    assert pts[-1][:2] == (2.0, 1.0)


def test_wait_for_nav_confirm_from_queue():
    q: queue.Queue[str] = queue.Queue()
    q.put("maybe")
    q.put("y")
    assert wait_for_nav_confirm("ok?", input_queue=q, timeout_s=2.0) is True


def test_wait_for_nav_confirm_reject():
    q: queue.Queue[str] = queue.Queue()
    q.put("n")
    assert wait_for_nav_confirm("ok?", input_queue=q, timeout_s=2.0) is False


def test_confirm_navigation_plan_noop_when_disabled():
    class _C:
        confirm_navigation = False

    assert confirm_navigation_plan(_C(), [[0, 0, 0], [1, 0, 0]]) is True


def test_confirm_navigation_plan_auto_yes():
    class _C:
        confirm_navigation = True
        _nav_confirm_auto_yes = True

    assert confirm_navigation_plan(_C(), [[0, 0, 0], [1, 0, 0]]) is True


def test_render_nav_plan_map_rgb_smoke():
    class _Grid:
        grid_origin = np.array([16.0, 16.0])
        grid_resolution = 0.1

    class _VM:
        grid_origin = np.array([16.0, 16.0])
        grid_resolution = 0.1

        def get_2d_map(self):
            obst = np.zeros((32, 32), dtype=bool)
            obst[10:12, 5:20] = True
            explored = np.zeros((32, 32), dtype=bool)
            explored[8:20, 4:22] = True
            return obst, explored

    img = render_nav_plan_map_rgb(
        _VM(),
        robot_xy=(0.0, 0.0),
        traj=[(0.0, 0.0, 0.0), (0.5, 0.2, 0.1), (1.0, 0.4, 0.2)],
        object_xy=(1.0, 0.4),
        max_side=128,
    )
    assert img is not None
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.dtype == np.uint8
