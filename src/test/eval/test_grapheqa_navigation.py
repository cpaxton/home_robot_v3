# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from types import SimpleNamespace

import numpy as np
import torch

from emet.eval.grapheqa_navigation import ObservedSpaceNavigation, base_xyt, predicted_camera_pose
from emet.eval.grapheqa_sensor import command_camera_pose
from emet.robots.footprint import CircularFootprint


def test_command_prior_uses_local_camera_heading_and_head_motion_preserves_base():
    camera = command_camera_pose(yaw_deg=90, tilt_deg=-30, sensor_height=1.5)
    forward = predicted_camera_pose(camera, "move_forward", 0.1)
    np.testing.assert_allclose(forward[:3, 3], [0, 0.1, 1.5], atol=1e-12)
    lowered = predicted_camera_pose(camera, "look_down", 5)
    np.testing.assert_allclose(base_xyt(lowered), base_xyt(camera), atol=1e-12)
    assert np.isclose(np.rad2deg(np.arcsin(lowered[2, 2])), -35)


def test_circular_footprint_is_rotation_invariant_and_includes_intersected_cells():
    footprint = CircularFootprint(0.1)
    expected = footprint.grid_cells(0.05, [0, 0, 0], [10, 10])
    np.testing.assert_array_equal(expected, footprint.grid_cells(0.05, [0, 0, np.pi / 4], [10, 10]))
    cells = set(map(tuple, expected))
    assert (12, 11) in cells
    assert (12, 12) not in cells
    assert len(cells) == 21
    shifted = footprint.grid_cells(0.05, [0.025, 0, 0], [10, 10])
    assert (13, 10) in set(map(tuple, shifted))


def test_frontiers_exclude_disconnected_observed_islands():
    nav = object.__new__(ObservedSpaceNavigation)
    nav.pose = np.zeros(3)
    explored = np.zeros((40, 40), dtype=bool)
    explored[3:20, 3:20] = True
    explored[25:35, 25:35] = True
    navigable = explored.copy()
    origin = torch.tensor([4.0, 4.0, 0.0])
    nav.map = SimpleNamespace(grid_origin=origin)
    nav._navigable = lambda: (navigable, explored)
    points = nav.frontiers()
    assert points and all(np.linalg.norm(point[:2]) < 1.0 for point in points)


def test_observed_space_plan_rejects_unobserved_goal_without_sweeping():
    nav = object.__new__(ObservedSpaceNavigation)
    nav.pose = np.zeros(3)
    nav.map = SimpleNamespace(grid_origin=torch.tensor([10.0, 10.0, 0.0]))
    free = np.ones((20, 20), dtype=bool)
    nav._navigable = lambda: (free, free)
    assert nav.plan([100, 100, 0], object_target=False) == ([], "no_valid_goals")
