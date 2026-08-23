# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Map ZMQ observations to graph viewpoint positions in MuJoCo / map world frame."""

from __future__ import annotations

from typing import Any

import numpy as np

from emet.utils.geometry import nav_xyt_to_world_xyt


def viewer_xyz_world_from_observation(
    obs: Any,
    *,
    robot: Any | None = None,
    floor_z: float = 0.0,
) -> np.ndarray | None:
    """
      Robot base ``(x, y, z)`` in the same world frame as fused depth / ``camera_pose``.

      Uses ``gps`` + ``compass`` + ``navigation_origin_xyt`` (same as DynaMem ``base_xyt`` and
    Rerun ``world/robot``). Raw :meth:`robot.get_base_pose` is episode-relative on Robocasa sim and
      must not be used alone for graph viewpoint nodes.
    """
    if obs is not None:
        gps = getattr(obs, "gps", None)
        compass = getattr(obs, "compass", None)
        if gps is not None and compass is not None:
            g = np.asarray(gps, dtype=np.float64).reshape(-1)
            c = np.asarray(compass, dtype=np.float64).ravel()
            if g.size >= 2 and c.size >= 1:
                local = np.array([float(g[0]), float(g[1]), float(c[0])], dtype=np.float64)
                sess = getattr(obs, "emet_session", None)
                wxyt = nav_xyt_to_world_xyt(local, sess)
                return np.array([float(wxyt[0]), float(wxyt[1]), float(floor_z)], dtype=np.float64)

    if robot is not None:
        try:
            bp = np.asarray(robot.get_base_pose(), dtype=np.float64).reshape(-1)
            if bp.size >= 2:
                bz = float(bp[2]) if bp.size >= 3 else float(floor_z)
                return np.array([float(bp[0]), float(bp[1]), bz], dtype=np.float64)
        except Exception:
            pass

    if obs is not None and getattr(obs, "camera_pose", None) is not None:
        cp = np.asarray(obs.camera_pose, dtype=np.float64)
        if cp.shape == (4, 4):
            return cp[:3, 3].copy()
    return None
