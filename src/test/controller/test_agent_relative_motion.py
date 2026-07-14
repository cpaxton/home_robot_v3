# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# Licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Relative base motion helpers for agent rotate_base / move_forward."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from emet.controller.controller_dynamem import DynamemController
from emet.agent.tools import get_tools


def test_rotate_base_and_move_forward_tools_registered():
    tools = {t.name: t for t in get_tools({})}
    assert "rotate_base" in tools
    assert "move_forward" in tools
    assert tools["rotate_base"].to_executor({"degrees": 180}) == [("rotate_base", "180")]
    assert tools["move_forward"].to_executor({"meters": 0.5}) == [("move_forward", "0.5")]


def test_clip_forward_distance_hits_obstacle():
    agent = DynamemController.__new__(DynamemController)
    agent.robot = SimpleNamespace(get_base_pose=lambda: np.array([0.0, 0.0, 0.0]))
    # 10x10 grid, obstacle at x≈0.3m if resolution 0.1 and origin at center-ish.
    # Simpler: mock xy_to_grid_coords and get_2d_map.
    obstacles = np.zeros((20, 20), dtype=bool)
    obstacles[10, 13] = True  # will map probe points here via mock

    class FakeVM:
        def is_empty(self):
            return False

        def get_2d_map(self):
            return obstacles, np.ones_like(obstacles)

        def xy_to_grid_coords(self, xy):
            x = float(np.asarray(xy).reshape(-1)[0])
            # Every 0.05m → +1 cell in j from 10
            j = 10 + int(round(x / 0.05))
            return np.array([10, j])

    agent.get_voxel_map = lambda: FakeVM()
    clipped = agent.clip_forward_distance_m(0.5, step_m=0.05)
    # Obstacle at j=13 → x=0.15; last free step before that is 0.10
    assert clipped == 0.10


def test_rotate_base_degrees_calls_relative_move():
    agent = DynamemController.__new__(DynamemController)
    agent._realtime_updates = True
    agent.announce_action = MagicMock()
    agent._find_phase_nav_timeout = lambda: 2.0
    moved = {}

    def move_base_to(xyt, **kwargs):
        moved["xyt"] = list(xyt)
        moved["kwargs"] = kwargs

    agent.robot = SimpleNamespace(
        move_to_nav_posture=lambda: None,
        move_base_to=move_base_to,
    )
    out = agent.rotate_base_degrees(180)
    assert out == 180.0
    assert moved["kwargs"]["relative"] is True
    assert abs(moved["xyt"][2] - np.pi) < 1e-6
