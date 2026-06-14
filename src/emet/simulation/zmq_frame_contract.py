# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""ZMQ observation frame contract checks (gps episode-relative, camera_pose MuJoCo world)."""

from __future__ import annotations

from typing import Any

import numpy as np

from emet.core.zmq_protocol import EMET_ZMQ_SESSION_KEY
from emet.utils.geometry import nav_xyt_to_world_xyt, pose_global_to_base

_SPAWN_ORIGIN_MIN_NORM_M = 0.5
_HEAD_BASE_MAX_XY_M = 1.5


def assert_camera_pose_is_mujoco_world(
    camera_pose: np.ndarray,
    *,
    navigation_origin_xyt: np.ndarray | list | None,
    gps: np.ndarray | list | None,
    compass: np.ndarray | list | None,
    atol: float = 1e-3,
) -> None:
    """Fail when ``camera_pose`` looks episode-relative while spawn origin is far from world zero."""
    cp = np.asarray(camera_pose, dtype=np.float64)
    if cp.shape != (4, 4) or not np.all(np.isfinite(cp)):
        raise AssertionError(f"camera_pose must be finite 4x4, got shape {cp.shape}")

    if navigation_origin_xyt is None:
        return
    org = np.asarray(navigation_origin_xyt, dtype=np.float64).reshape(-1)[:3]
    if float(np.linalg.norm(org[:2])) < _SPAWN_ORIGIN_MIN_NORM_M:
        return

    g = np.asarray(gps if gps is not None else [0.0, 0.0], dtype=np.float64).reshape(-1)[:2]
    c = np.asarray(compass if compass is not None else [0.0], dtype=np.float64).ravel()
    if float(np.linalg.norm(g[:2])) > 0.15 or (c.size and abs(float(c[0])) > 0.15):
        return

    cam_xy = cp[:2, 3]
    if float(np.linalg.norm(cam_xy)) < _SPAWN_ORIGIN_MIN_NORM_M:
        raise AssertionError(
            "camera_pose translation looks episode-relative at spawn "
            f"(cam_xy={cam_xy.tolist()}, navigation_origin_xyt={org.tolist()})"
        )


def assert_zmq_observation_frames_consistent(
    msg: dict[str, Any],
    *,
    atol: float = 0.25,
) -> None:
    """``gps``/``compass``/``navigation_origin_xyt``/``camera_pose`` share one world frame."""
    sess = msg.get(EMET_ZMQ_SESSION_KEY)
    if not isinstance(sess, dict):
        sess = {}
    org = sess.get("navigation_origin_xyt")

    gps = msg.get("gps")
    compass = msg.get("compass")
    cp = msg.get("camera_pose")
    if cp is None:
        raise AssertionError("camera_pose missing from observation")

    assert_camera_pose_is_mujoco_world(
        np.asarray(cp, dtype=np.float64),
        navigation_origin_xyt=org,
        gps=gps,
        compass=compass,
    )

    if gps is None or compass is None:
        return

    g = np.asarray(gps, dtype=np.float64).reshape(-1)
    c = np.asarray(compass, dtype=np.float64).ravel()
    if g.size < 2 or c.size < 1:
        return

    local = np.array([float(g[0]), float(g[1]), float(c[0])], dtype=np.float64)
    base_w = nav_xyt_to_world_xyt(local, sess)
    cam_t = np.asarray(cp, dtype=np.float64)[:3, 3]
    dxy = float(np.linalg.norm(cam_t[:2] - base_w[:2]))
    if dxy > _HEAD_BASE_MAX_XY_M:
        raise AssertionError(
            f"camera_pose XY too far from nav base at spawn (dxy={dxy:.3f}m, "
            f"base_w={base_w[:2].tolist()}, cam_t={cam_t[:2].tolist()}, origin={org})"
        )


def episode_relative_camera_pose(world_pose: np.ndarray, navigation_origin_xyt: np.ndarray) -> np.ndarray:
    """Helper for regression tests: what wrongly publishing episode-relative looks like."""
    return pose_global_to_base(np.asarray(world_pose, dtype=np.float64), navigation_origin_xyt)
