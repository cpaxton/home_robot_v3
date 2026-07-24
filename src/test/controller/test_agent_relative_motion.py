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
import pytest

from emet.agent.tools import get_tools
from emet.controller.controller_dynamem import DynamemController


def test_rotate_base_and_move_forward_tools_registered():
    tools = {t.name: t for t in get_tools({})}
    assert "rotate_base" in tools
    assert "move_forward" in tools
    # Relative motion tools run via func (map-aware), not executor_commands mapping.
    assert tools["rotate_base"].to_executor({"degrees": 180}) == []
    assert tools["move_forward"].to_executor({"meters": 0.5}) == []
    assert tools["move_forward"].returns_info is True


def test_clip_forward_distance_hits_obstacle():
    agent = DynamemController.__new__(DynamemController)
    agent.robot = SimpleNamespace(get_base_pose=lambda: np.array([0.0, 0.0, 0.0]))
    obstacles = np.zeros((20, 20), dtype=bool)
    obstacles[10, 13] = True  # will map probe points here via mock
    explored = np.ones((20, 20), dtype=bool)

    class FakeVM:
        def is_empty(self):
            return False

        def get_2d_map(self):
            return obstacles, explored

        def xy_to_grid_coords(self, xy):
            x = float(np.asarray(xy).reshape(-1)[0])
            # Every 0.05m → +1 cell in j from 10
            j = 10 + int(round(x / 0.05))
            return np.array([10, j])

    agent.get_voxel_map = lambda: FakeVM()
    clipped = agent.clip_forward_distance_m(0.5, step_m=0.05, clearance_m=0.05)
    # Obstacle at j=13 → x=0.15; last free step 0.10, minus 0.05 clearance → 0.05
    assert clipped == pytest.approx(0.05)


def test_clip_forward_refuses_empty_map_without_seed():
    agent = DynamemController.__new__(DynamemController)
    agent.robot = SimpleNamespace(get_base_pose=lambda: np.array([0.0, 0.0, 0.0]))

    class EmptyVM:
        def is_empty(self):
            return True

        def get_2d_map(self):
            return np.zeros((8, 8), dtype=bool), np.zeros((8, 8), dtype=bool)

    agent.get_voxel_map = lambda: EmptyVM()
    assert agent.clip_forward_distance_m(0.1) == 0.0

    class BlankVM:
        def is_empty(self):
            return False

        def get_2d_map(self):
            z = np.zeros((8, 8), dtype=bool)
            return z, z

        def xy_to_grid_coords(self, xy):
            return np.array([0, 0])

    agent.get_voxel_map = lambda: BlankVM()
    assert agent.clip_forward_distance_m(0.1) == 0.0


def test_clip_forward_seeds_local_radius_disk():
    """Empty cloud + _update_visited → short nudge inside explored disk, not beyond."""
    agent = DynamemController.__new__(DynamemController)
    agent.robot = SimpleNamespace(get_base_pose=lambda: np.array([0.0, 0.0, 0.0]))
    obstacles = np.zeros((40, 40), dtype=bool)
    explored = np.zeros((40, 40), dtype=bool)
    seeded = {"done": False}

    class SeedableVM:
        def is_empty(self):
            return True

        def _update_visited(self, pose):
            # Disk around cell (20, 20): ±4 cells (~0.2 m at 0.05 m/cell in this mock).
            explored[16:25, 16:25] = True
            seeded["done"] = True

        def get_2d_map(self):
            return obstacles, explored

        def xy_to_grid_coords(self, xy):
            x = float(np.asarray(xy).reshape(-1)[0])
            j = 20 + int(round(x / 0.05))
            return np.array([20, j])

    agent.get_voxel_map = lambda: SeedableVM()
    clipped = agent.clip_forward_distance_m(0.5, step_m=0.05, clearance_m=0.0)
    assert seeded["done"] is True
    # Explored j in [16, 24]; from j=20 forward max j=24 → 0.20 m
    assert clipped == pytest.approx(0.20)


def test_move_forward_tool_asks_when_cannot_drive():
    agent = MagicMock()
    agent.move_forward_meters = MagicMock(return_value=0.0)
    executor = MagicMock()
    executor.agent = agent
    tools_ctx = {t.name: t for t in get_tools({"executor": executor})}
    msg = tools_ctx["move_forward"].func(meters=0.3)
    assert "scan" in msg.lower() or "rotate" in msg.lower()
    assert "?" in msg


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
