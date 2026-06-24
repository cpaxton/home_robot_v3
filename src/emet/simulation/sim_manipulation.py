# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Sim-only pick/place helpers (teleport freejoint bodies for OVMM full-task benchmark)."""

from __future__ import annotations

import time
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


def set_named_joint_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    value: float,
) -> bool:
    """Set qpos of a named hinge/slide joint (cabinet doors, drawers); zeros its qvel."""
    try:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name))
    except Exception:
        return False
    if joint_id < 0:
        return False
    jnt_type = int(model.jnt_type[joint_id])
    if jnt_type not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
        return False
    v = float(value)
    if model.jnt_limited[joint_id]:
        lo, hi = float(model.jnt_range[joint_id][0]), float(model.jnt_range[joint_id][1])
        v = min(max(v, lo), hi)
    data.qpos[int(model.jnt_qposadr[joint_id])] = v
    data.qvel[int(model.jnt_dofadr[joint_id])] = 0.0
    mujoco.mj_forward(model, data)
    return True


def get_named_joint_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
) -> float | None:
    """Read back qpos of a named hinge/slide joint; ``None`` if missing or wrong type."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name))
    if joint_id < 0:
        return None
    jnt_type = int(model.jnt_type[joint_id])
    if jnt_type not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
        return None
    return float(data.qpos[int(model.jnt_qposadr[joint_id])])


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


def robot_zmq_set_body_pose(
    robot: Any,
    body: str,
    pos: np.ndarray | list[float],
    *,
    quat: np.ndarray | list[float] | None = None,
) -> None:
    """Send ``sim_set_body_pose`` on the robot ZMQ client (OVMM full + dynamic exploration)."""
    from emet.core.zmq_protocol import build_sim_set_body_pose_action

    p = np.asarray(pos, dtype=np.float64).reshape(3)
    step = int(getattr(robot, "_last_step", -1)) + 1
    if step < 1:
        step = 1
    quat_arg = None
    if quat is not None:
        quat_arg = [float(x) for x in np.asarray(quat, dtype=np.float64).reshape(4)]
    action = build_sim_set_body_pose_action(step, body, p.tolist(), quat=quat_arg)
    _send_meta_action(robot, action)


def parse_sim_set_joint_qpos_action(raw: Any) -> tuple[str | None, float | None]:
    """Parse ``sim_set_joint_qpos`` recv action payload."""
    if not isinstance(raw, dict):
        return None, None
    joint = raw.get("joint")
    value = raw.get("value")
    if not joint or value is None:
        return None, None
    return str(joint), float(value)


def robot_zmq_set_joint_qpos(robot: Any, joint: str, value: float) -> None:
    """Send ``sim_set_joint_qpos`` on the robot ZMQ client (doors/drawers in dynamic benchmarks)."""
    from emet.core.zmq_protocol import build_sim_set_joint_qpos_action

    step = int(getattr(robot, "_last_step", -1)) + 1
    if step < 1:
        step = 1
    action = build_sim_set_joint_qpos_action(step, joint, float(value))
    _send_meta_action(robot, action)


def _send_meta_action(robot: Any, action: dict[str, Any]) -> None:
    """Send a meta action reliably and give the sim a beat to apply it."""
    send_action = getattr(robot, "send_action", None)
    if callable(send_action):
        send_action(action, reliable=True)
    else:
        robot.send_message(action)
    wait_obs = getattr(robot, "wait_for_obs", None)
    if callable(wait_obs):
        wait_obs(timeout=5.0)
    time.sleep(0.25)
