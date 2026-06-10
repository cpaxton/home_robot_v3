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

"""Sim-only pick/place helpers (teleport freejoint bodies for OVMM full-task benchmark)."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


def set_free_body_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    pos: np.ndarray | list[float],
    quat: np.ndarray | list[float] | None = None,
) -> bool:
    """Set world pose of a body with a single free joint (pos + optional wxyz quat)."""
    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(body_name))
    except Exception:
        return False
    if body_id < 0:
        return False
    jnt_adr = int(model.body_jntadr[body_id])
    if jnt_adr < 0 or int(model.jnt_type[jnt_adr]) != int(mujoco.mjtJoint.mjJNT_FREE):
        return False
    qpos_adr = int(model.jnt_qposadr[jnt_adr])
    p = np.asarray(pos, dtype=np.float64).reshape(3)
    data.qpos[qpos_adr : qpos_adr + 3] = p
    if quat is not None:
        q = np.asarray(quat, dtype=np.float64).reshape(4)
    else:
        q = np.array(data.qpos[qpos_adr + 3 : qpos_adr + 7], dtype=np.float64)
        if not np.isfinite(q).all() or abs(float(np.linalg.norm(q)) - 1.0) > 1e-3:
            q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = q
    data.qvel[qpos_adr : qpos_adr + 6] = 0.0
    mujoco.mj_forward(model, data)
    return True


def parse_sim_set_body_pose_action(raw: Any) -> tuple[str | None, list[float] | None, list[float] | None]:
    """Parse ``sim_set_body_pose`` recv action payload."""
    if not isinstance(raw, dict):
        return None, None, None
    body = raw.get("body")
    pos = raw.get("pos")
    if not body or pos is None:
        return None, None, None
    pos_list = [float(x) for x in np.asarray(pos, dtype=np.float64).reshape(-1)[:3]]
    quat_raw = raw.get("quat")
    quat_list = None
    if quat_raw is not None:
        quat_list = [float(x) for x in np.asarray(quat_raw, dtype=np.float64).reshape(-1)[:4]]
    return str(body), pos_list, quat_list
