# Copyright (c) Chris Paxton 2026
"""Unit tests for A* start-escape ring config (no GPU)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from emet.motion.algo.a_star import AStar


def _planner_with_map(obs: np.ndarray, *, start_escape_max_ring: int) -> AStar:
    space = MagicMock()
    space.is_valid = lambda *a, **k: True
    space.voxel_map.get_2d_map.return_value = (obs, np.ones_like(obs, dtype=bool))
    return AStar(space, start_escape_max_ring=start_escape_max_ring)


def test_start_escape_max_ring_finds_cell_beyond_default_four():
    # 15x15: start at center occupied; free cell only at Chebyshev distance 6.
    n = 15
    obs = np.ones((n, n), dtype=bool)
    c = n // 2
    obs[c + 6, c] = False
    planner = _planner_with_map(obs, start_escape_max_ring=8)
    found = planner.get_unoccupied_neighbor((c, c), max_ring=planner.start_escape_max_ring)
    assert found == (c + 6, c)


def test_start_escape_respects_small_configured_ring():
    n = 15
    obs = np.ones((n, n), dtype=bool)
    c = n // 2
    obs[c + 6, c] = False
    planner = _planner_with_map(obs, start_escape_max_ring=4)
    found = planner.get_unoccupied_neighbor((c, c), max_ring=planner.start_escape_max_ring)
    assert found is None
