# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""KinematicPickPlace must sync the MJCF freejoint in MuJoCo world, not episode GPS."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np


def test_world_base_xyt_composes_navigation_origin():
    from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor
    from emet.utils.geometry import xyt_base_to_global

    origin = np.array([1.5, 1.6, 0.1], dtype=np.float64)
    episode = np.array([-1.2, -1.8, -np.pi / 2], dtype=np.float64)
    expected = xyt_base_to_global(episode, origin)

    robot = MagicMock()
    robot.get_base_pose.return_value = episode
    robot.get_emet_session.return_value = {"navigation_origin_xyt": origin.tolist()}
    robot._state = {"base_xyz": [float(expected[0]), float(expected[1]), 0.42]}

    exe = object.__new__(KinematicPickPlaceExecutor)
    exe.robot = robot
    world = exe._world_base_xyt()
    assert world is not None
    np.testing.assert_allclose(world[:2], expected[:2], atol=1e-6)
    np.testing.assert_allclose(world[2], expected[2], atol=1e-6)


def test_world_base_xyt_prefers_base_xyz_xy():
    from emet.controller.manipulation.kinematic_pick_place import KinematicPickPlaceExecutor

    robot = MagicMock()
    robot.get_base_pose.return_value = np.array([0.0, 0.0, -1.57])
    robot.get_emet_session.return_value = {
        "navigation_origin_xyt": [1.0, 2.0, 0.0],
    }
    # World XY from MuJoCo body (authoritative); yaw still from composed GPS.
    robot._state = {"base_xyz": [0.273, -0.203, 0.55]}

    exe = object.__new__(KinematicPickPlaceExecutor)
    exe.robot = robot
    world = exe._world_base_xyt()
    assert world is not None
    np.testing.assert_allclose(world[:2], [0.273, -0.203], atol=1e-6)
