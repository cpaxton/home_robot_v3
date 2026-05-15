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

"""Heuristic checks that the MuJoCo ZMQ sim has not lost stability (robot ``flying away``, NaNs)."""

from __future__ import annotations

from typing import Any

import numpy as np


class RobotSimPhysicsExplodedError(RuntimeError):
    """Raised when streamed robot state looks like a physics blow-up (not a normal navigation pose)."""


# Mixed joint vector can include m/s and rad/s; free-flyers that explode often hit huge linear rates.
_MAX_SINGLE_JOINT_SPEED: float = 85.0
_MAX_JOINT_SPEED_L2: float = 220.0
# Head camera in the default table scene should stay in a bounded volume; a launched base sends it sky-high.
_MAX_CAMERA_ORIGIN_NORM: float = 26.0
_MAX_CAMERA_Z_ABS: float = 12.0
# World-frame GPS on Stretch (m); default table is sub-room scale.
_MAX_GPS_XY_NORM: float = 18.0
# Robosuite ``base_pose`` in state is nav-relative SE(2); huge values usually mean the sim diverged.
_MAX_BASE_POSE_XY_NORM: float = 14.0


def check_robot_sim_stable(robot: Any, *, stage: str) -> None:
    """Raise :class:`RobotSimPhysicsExplodedError` if the last ZMQ state/obs looks unstable.

    Intended for dataset capture and similar scripts; safe to call whenever ``get_joint_state`` /
    ``get_observation`` are valid.
    """
    reasons: list[str] = []

    if hasattr(robot, "get_joint_state"):
        js = robot.get_joint_state(timeout=0.5)
        if js is not None and js[0] is not None:
            q, dq, _tau = js
            q = np.asarray(q, dtype=np.float64).reshape(-1)
            dq = np.asarray(dq, dtype=np.float64).reshape(-1)
            if not np.isfinite(q).all():
                reasons.append("joint_positions contain NaN/Inf")
            if dq.size and not np.isfinite(dq).all():
                reasons.append("joint_velocities contain NaN/Inf")
            if dq.size:
                mx = float(np.max(np.abs(dq)))
                l2 = float(np.linalg.norm(dq))
                if mx > _MAX_SINGLE_JOINT_SPEED:
                    reasons.append(f"max|joint_velocity|={mx:.1f} (threshold {_MAX_SINGLE_JOINT_SPEED})")
                if l2 > _MAX_JOINT_SPEED_L2:
                    reasons.append(f"||joint_velocity||_2={l2:.1f} (threshold {_MAX_JOINT_SPEED_L2})")

    if hasattr(robot, "get_observation"):
        try:
            obs = robot.get_observation()
        except Exception:
            obs = None
        if obs is not None and getattr(obs, "camera_pose", None) is not None:
            M = np.asarray(obs.camera_pose, dtype=np.float64).reshape(4, 4)
            if M.shape == (4, 4) and np.isfinite(M).all():
                t = M[:3, 3]
                n = float(np.linalg.norm(t))
                if n > _MAX_CAMERA_ORIGIN_NORM:
                    reasons.append(f"||camera_origin||={n:.2f} m (threshold {_MAX_CAMERA_ORIGIN_NORM})")
                if abs(float(t[2])) > _MAX_CAMERA_Z_ABS:
                    reasons.append(f"|camera_z|={abs(float(t[2])):.2f} m (threshold {_MAX_CAMERA_Z_ABS})")

    _check_zmq_dicts(robot, reasons)

    if reasons:
        msg = "; ".join(reasons)
        raise RobotSimPhysicsExplodedError(f"[{stage}] unstable sim / robot state: {msg}")


def _check_zmq_dicts(robot: Any, reasons: list[str]) -> None:
    """Use raw ZMQ observation for GPS; state dict for base_pose."""
    lock = getattr(robot, "_obs_lock", None)
    obs: dict[str, Any] | None = None
    state: dict[str, Any] | None = None
    if lock is None:
        obs = getattr(robot, "_obs", None)
        state = getattr(robot, "_state", None)
    else:
        with lock:
            obs = getattr(robot, "_obs", None)
            state = getattr(robot, "_state", None)

    if isinstance(obs, dict):
        gps = obs.get("gps")
        if gps is not None:
            g = np.asarray(gps, dtype=np.float64).reshape(-1)
            if g.size >= 2 and np.isfinite(g).all():
                gn = float(np.hypot(g[0], g[1]))
                if gn > _MAX_GPS_XY_NORM:
                    reasons.append(f"||gps_xy||={gn:.2f} m (threshold {_MAX_GPS_XY_NORM})")

    if isinstance(state, dict):
        bp = state.get("base_pose")
        if bp is not None:
            b = np.asarray(bp, dtype=np.float64).reshape(-1)
            if b.size >= 2 and np.isfinite(b).all():
                bn = float(np.hypot(b[0], b[1]))
                if bn > _MAX_BASE_POSE_XY_NORM:
                    reasons.append(f"||base_pose_xy||={bn:.2f} (threshold {_MAX_BASE_POSE_XY_NORM})")
