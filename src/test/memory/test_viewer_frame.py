# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from emet.memory.graph_eqa.viewer_frame import viewer_xyz_world_from_observation


def test_viewer_xyz_uses_navigation_origin_not_episode_relative_base():
    """World viewpoint must match gps+origin (same frame as camera / voxel map)."""
    obs = SimpleNamespace(
        gps=np.array([1.0, 0.5]),
        compass=np.array([0.25]),
        emet_session={"navigation_origin_xyt": [10.0, 20.0, 0.0]},
        camera_pose=np.eye(4),
    )
    robot = SimpleNamespace(get_base_pose=lambda: np.array([1.0, 0.5, 0.0]))
    v = viewer_xyz_world_from_observation(obs, robot=robot)
    assert v is not None
    # local (1, 0.5) composed with origin (10, 20) -> ~(11, 20.5) in world (exact depends on rotation)
    assert abs(float(v[0]) - 11.0) < 0.05
    assert abs(float(v[1]) - 20.5) < 0.05
    assert float(v[2]) == 0.0


def test_viewer_xyz_falls_back_to_robot_without_gps():
    robot = SimpleNamespace(get_base_pose=lambda: np.array([3.0, 4.0, 0.1]))
    v = viewer_xyz_world_from_observation(SimpleNamespace(), robot=robot)
    np.testing.assert_allclose(v, [3.0, 4.0, 0.1])
