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
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

"""MuJoCo MJCF → Rerun: articulated robot skeleton for GenericZmqClient / RobotSpec."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from emet.core.zmq_protocol import EMET_ZMQ_SESSION_KEY
from emet.utils.geometry import xyt_base_to_global

_MAX_BODIES = 72


def _quat_wxyz_yaw(theta: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(theta)
    return float(np.cos(half)), 0.0, 0.0, float(np.sin(half))


def _safe_entity_segment(name: str) -> str:
    t = re.sub(r"[^0-9A-Za-z_.-]+", "_", (name or "").strip())
    return t or "body"


def _base_freejoint_qadr(model: Any, base_link_name: str) -> int | None:
    import mujoco as mj

    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, base_link_name)
    if bid < 0:
        return None
    for j in range(model.njnt):
        if int(model.jnt_bodyid[j]) != bid:
            continue
        if int(model.jnt_type[j]) != mj.mjtJoint.mjJNT_FREE:
            continue
        return int(model.jnt_qposadr[j])
    return None


def _subtree_body_ids(model: Any, root_bid: int) -> list[int]:
    children: list[list[int]] = [[] for _ in range(model.nbody)]
    for b in range(1, model.nbody):
        p = int(model.body_parentid[b])
        if 0 <= p < model.nbody:
            children[p].append(b)
    out: list[int] = []
    stack = [root_bid]
    while stack:
        b = stack.pop()
        out.append(b)
        stack.extend(children[b])
    return out


class MjcfBodySkeletonLogger:
    """Drive a local copy of the sim MJCF from ZMQ-style obs and log body frames to Rerun."""

    def __init__(self, mjcf_path: str | Path, joint_names: tuple[str, ...] | list[str], dof: int, base_link_name: str):
        import mujoco

        path = Path(mjcf_path)
        if not path.is_file():
            raise FileNotFoundError(f"MJCF not found: {path}")
        self._mjcf_path = str(path.resolve())
        self.model = mujoco.MjModel.from_xml_path(self._mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.joint_names = tuple(joint_names)
        self._dof = int(dof)
        self.base_link_name = str(base_link_name)
        self._free_qadr = _base_freejoint_qadr(self.model, self.base_link_name)
        root_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.base_link_name)
        if root_bid < 0:
            raise ValueError(f"base_link body {self.base_link_name!r} not found in {self._mjcf_path}")
        bodies = _subtree_body_ids(self.model, int(root_bid))
        self._body_ids = bodies[:_MAX_BODIES]
        self._body_paths: list[str] = []
        for bid in self._body_ids:
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body{bid}"
            self._body_paths.append(f"world/robot/mjcf/{_safe_entity_segment(nm)}")
        self._nav_origin: np.ndarray | None = None

    def apply_and_log(self, obs_pose: dict[str, Any]) -> None:
        import mujoco
        import rerun as rr

        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0

        if self._nav_origin is None:
            sess = obs_pose.get(EMET_ZMQ_SESSION_KEY)
            if isinstance(sess, dict):
                org = sess.get("navigation_origin_xyt")
                if org is not None:
                    self._nav_origin = np.asarray(org, dtype=np.float64).reshape(-1)[:3].copy()

        if self._free_qadr is not None:
            qadr = self._free_qadr
            gps = np.asarray(obs_pose.get("gps", np.zeros(2)), dtype=np.float64).reshape(-1)[:2]
            comp = np.asarray(obs_pose.get("compass", np.zeros(1)), dtype=np.float64).ravel()
            theta = float(comp[0]) if comp.size else 0.0
            local_xyt = np.array([float(gps[0]), float(gps[1]), theta], dtype=np.float64)
            if self._nav_origin is not None:
                world_xyt = xyt_base_to_global(local_xyt, self._nav_origin)
            else:
                world_xyt = local_xyt
            z0 = float(self.model.qpos0[qadr + 2])
            qw, qx, qy, qz = _quat_wxyz_yaw(float(world_xyt[2]))
            self.data.qpos[qadr : qadr + 3] = [float(world_xyt[0]), float(world_xyt[1]), z0]
            self.data.qpos[qadr + 3 : qadr + 7] = [qw, qx, qy, qz]

        jraw = obs_pose.get("joint")
        if jraw is None:
            jvec = np.zeros(max(self._dof, len(self.joint_names)), dtype=np.float64)
        else:
            jvec = np.asarray(jraw, dtype=np.float64).reshape(-1)

        for i, jname in enumerate(self.joint_names):
            if i >= len(jvec):
                break
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                continue
            jt = int(self.model.jnt_type[jid])
            qadr = int(self.model.jnt_qposadr[jid])
            if jt == mujoco.mjtJoint.mjJNT_FREE:
                continue
            if jt == mujoco.mjtJoint.mjJNT_HINGE or jt == mujoco.mjtJoint.mjJNT_SLIDE:
                self.data.qpos[qadr] = float(jvec[i])

        mujoco.mj_forward(self.model, self.data)

        for bid, path in zip(self._body_ids, self._body_paths, strict=True):
            b = self.data.body(bid)
            R = np.asarray(b.xmat, dtype=np.float64).reshape(3, 3)
            p = np.asarray(b.xpos, dtype=np.float64).reshape(3)
            rr.log(path, rr.Transform3D(translation=p, mat3x3=R, axis_length=0.07))
