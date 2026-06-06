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

from emet.core.zmq_protocol import EMET_ZMQ_SESSION_KEY, read_emet_session
from emet.utils.geometry import nav_xyt_to_world_xyt, xyt_base_to_global

_MAX_BODIES = 72


def apply_zmq_obs_to_mujoco_data(
    model: Any,
    data: Any,
    obs_pose: dict[str, Any],
    *,
    joint_names: tuple[str, ...],
    dof: int,
    base_link_name: str,
    nav_origin_slot: list[np.ndarray | None],
    free_qadr: int | None,
) -> None:
    """Reset ``data.qpos`` to defaults, then fill from ZMQ-style ``gps``/``compass``/``joint``.

    Planar slide/slide/yaw values in ``joint`` are replayed as-is. Absolute body poses in the
    standalone MJCF sit in that model's world (often ``base_root`` at origin); callers align Rerun
    with the sim by logging **base_link-relative** transforms under ``world/robot`` (see
    :meth:`MjcfBodySkeletonLogger.apply_and_log`).
    """
    import mujoco

    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0

    if nav_origin_slot[0] is None:
        sess = obs_pose.get(EMET_ZMQ_SESSION_KEY)
        if isinstance(sess, dict):
            org = sess.get("navigation_origin_xyt")
            if org is not None:
                nav_origin_slot[0] = np.asarray(org, dtype=np.float64).reshape(-1)[:3].copy()

    if free_qadr is not None:
        qadr = free_qadr
        gps = np.asarray(obs_pose.get("gps", np.zeros(2)), dtype=np.float64).reshape(-1)[:2]
        comp = np.asarray(obs_pose.get("compass", np.zeros(1)), dtype=np.float64).ravel()
        theta = float(comp[0]) if comp.size else 0.0
        local_xyt = np.array([float(gps[0]), float(gps[1]), theta], dtype=np.float64)
        if nav_origin_slot[0] is not None:
            world_xyt = xyt_base_to_global(local_xyt, nav_origin_slot[0])
        else:
            world_xyt = local_xyt
        z0 = float(model.qpos0[qadr + 2])
        qw, qx, qy, qz = _quat_wxyz_yaw(float(world_xyt[2]))
        data.qpos[qadr : qadr + 3] = [float(world_xyt[0]), float(world_xyt[1]), z0]
        data.qpos[qadr + 3 : qadr + 7] = [qw, qx, qy, qz]

    jraw = obs_pose.get("joint_mjcf")
    if jraw is None:
        jraw = obs_pose.get("joint")
    if jraw is None:
        jvec = np.zeros(max(dof, len(joint_names)), dtype=np.float64)
    else:
        jvec = np.asarray(jraw, dtype=np.float64).reshape(-1)
    if jvec.size != len(joint_names) and jvec.size == 11 and joint_names and joint_names[0] == "joint_lift":
        from emet.robots.stretch.joint_layout import robocasa_mjcf_joint_positions_from_hello_stretch

        mapped = robocasa_mjcf_joint_positions_from_hello_stretch(jvec)
        if mapped is not None:
            jvec = mapped

    for i, jname in enumerate(joint_names):
        if i >= len(jvec):
            break
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        jt = int(model.jnt_type[jid])
        qadr = int(model.jnt_qposadr[jid])
        if jt == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if jt == mujoco.mjtJoint.mjJNT_HINGE or jt == mujoco.mjtJoint.mjJNT_SLIDE:
            data.qpos[qadr] = float(jvec[i])


def _body_T_world(data: Any, bid: int) -> np.ndarray:
    b = data.body(bid)
    R = np.asarray(b.xmat, dtype=np.float64).reshape(3, 3)
    p = np.asarray(b.xpos, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _T_world_from_planar_xyt(x: float, y: float, yaw: float, z: float) -> np.ndarray:
    c, s = np.cos(float(yaw)), np.sin(float(yaw))
    T = np.eye(4, dtype=np.float64)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    T[0, 3], T[1, 3], T[2, 3] = float(x), float(y), float(z)
    return T


def _nav_world_xyt_from_obs(obs_pose: dict[str, Any]) -> np.ndarray:
    xy = np.asarray(obs_pose.get("gps", np.zeros(2)), dtype=np.float64).reshape(-1)[:2]
    comp = np.asarray(obs_pose.get("compass", np.zeros(1)), dtype=np.float64).ravel()
    theta = float(comp[0]) if comp.size else 0.0
    local = np.array([float(xy[0]), float(xy[1]), theta], dtype=np.float64)
    return nav_xyt_to_world_xyt(local, read_emet_session(obs_pose))


def _world_alignment_fixup_T(nav_wxyt: np.ndarray, T_standalone_base: np.ndarray) -> np.ndarray:
    """``T_nav_world @ inv(T_standalone_base)`` maps standalone-world points → nav / map world."""
    z = float(T_standalone_base[2, 3])
    T_nav = _T_world_from_planar_xyt(float(nav_wxyt[0]), float(nav_wxyt[1]), float(nav_wxyt[2]), z)
    return T_nav @ np.linalg.inv(T_standalone_base)


def _world_xyt_from_base_body(model: Any, data: Any, base_link_name: str) -> np.ndarray:
    """Planar (x, y, yaw) from ``base_link`` after ``mj_forward`` (matches sim ``get_base_xyt``)."""
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_link_name)
    if bid < 0:
        return np.zeros(3, dtype=np.float64)
    xpos = np.asarray(data.body(bid).xpos, dtype=np.float64).reshape(3)
    xmat = np.asarray(data.body(bid).xmat, dtype=np.float64).reshape(3, 3)
    theta = float(np.arctan2(xmat[1, 0], xmat[0, 0]))
    return np.array([float(xpos[0]), float(xpos[1]), theta], dtype=np.float64)


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
        import rerun as rr

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
        self._nav_origin_slot: list[np.ndarray | None] = [None]
        # Establish hierarchy: transforms on ``world/robot/mjcf/*`` are **relative to base_link**
        # (parent chain → ``world/robot`` uses the same nav-world pose as ``log_robot_xyt``).
        rr.log(
            "world/robot/mjcf",
            rr.Transform3D(translation=[0.0, 0.0, 0.0], mat3x3=np.eye(3)),
            static=True,
        )

    def apply_and_log(self, obs_pose: dict[str, Any]) -> None:
        import mujoco
        import rerun as rr

        apply_zmq_obs_to_mujoco_data(
            self.model,
            self.data,
            obs_pose,
            joint_names=self.joint_names,
            dof=self._dof,
            base_link_name=self.base_link_name,
            nav_origin_slot=self._nav_origin_slot,
            free_qadr=self._free_qadr,
        )

        mujoco.mj_forward(self.model, self.data)

        root_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.base_link_name)
        T_base_w = _body_T_world(self.data, int(root_bid))
        T_inv = np.linalg.inv(T_base_w)

        for bid, entity_path in zip(self._body_ids, self._body_paths, strict=True):
            T_w = _body_T_world(self.data, int(bid))
            T_rel = T_inv @ T_w
            R = T_rel[:3, :3]
            p = T_rel[:3, 3]
            rr.log(entity_path, rr.Transform3D(translation=p, mat3x3=R, axis_length=0.07))


class MjcfVisualMeshLogger:
    """Drive standalone MJCF from ZMQ obs; log visual meshes for Rerun."""

    def __init__(self, mjcf_path: str | Path, joint_names: tuple[str, ...] | list[str], dof: int, base_link_name: str):
        import mujoco

        path = Path(mjcf_path)
        if not path.is_file():
            raise FileNotFoundError(f"MJCF not found: {path}")
        self.model = mujoco.MjModel.from_xml_path(str(path.resolve()))
        self.data = mujoco.MjData(self.model)
        self.joint_names = tuple(joint_names)
        self._dof = int(dof)
        self.base_link_name = str(base_link_name)
        self._free_qadr = _base_freejoint_qadr(self.model, self.base_link_name)
        self._nav_origin_slot: list[np.ndarray | None] = [None]
        self._geom_mesh_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for gid in range(self.model.ngeom):
            if int(self.model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            mid = int(self.model.geom_dataid[gid])
            if mid < 0:
                continue
            vadr = int(self.model.mesh_vertadr[mid])
            vnum = int(self.model.mesh_vertnum[mid])
            verts = self.model.mesh_vert[vadr * 3 : (vadr + vnum) * 3].reshape(-1, 3).astype(np.float64).copy()
            fadr = int(self.model.mesh_faceadr[mid])
            fnum = int(self.model.mesh_facenum[mid])
            faces = self.model.mesh_face[fadr : fadr + fnum].reshape(-1, 3).astype(np.int32).copy()
            self._geom_mesh_cache[gid] = (verts, faces)

    def sync_kinematics(self, obs_pose: dict[str, Any]) -> np.ndarray:
        """Apply *obs_pose* to the local MJCF and return ``base_link`` world ``(x, y, yaw)``."""
        import mujoco

        apply_zmq_obs_to_mujoco_data(
            self.model,
            self.data,
            obs_pose,
            joint_names=self.joint_names,
            dof=self._dof,
            base_link_name=self.base_link_name,
            nav_origin_slot=self._nav_origin_slot,
            free_qadr=self._free_qadr,
        )
        mujoco.mj_forward(self.model, self.data)
        return _world_xyt_from_base_body(self.model, self.data, self.base_link_name)

    def log_meshes_world(self, rr: Any, obs_pose: dict[str, Any], *, entity_prefix: str = "da3/robot/mesh") -> None:
        """Log mesh geoms for Rerun.

        Under ``world/robot/…``: vertices are **base_link-relative** so ``world/robot`` (GPS + nav origin, set by
        :meth:`emet.visualization.rerun.RerunVisualizer.log_robot_xyt`) places the mesh in map world. That is
        equivalent to applying ``T_nav @ inv(T_standalone_base)`` in world space without moving the robot frame.

        Other prefixes: vertices stay in the standalone MJCF world frame (debug / DA3 overlays).
        """
        import mujoco

        self.sync_kinematics(obs_pose)
        root_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.base_link_name)
        T_base_w = _body_T_world(self.data, int(root_bid))
        T_inv = np.linalg.inv(T_base_w)
        use_base_relative = entity_prefix.startswith("world/robot")

        for gid, (V_loc, F) in self._geom_mesh_cache.items():
            R = np.asarray(self.data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
            p = np.asarray(self.data.geom_xpos[gid], dtype=np.float64).reshape(3)
            V_w = (R @ V_loc.T).T + p
            if use_base_relative:
                V_h = np.c_[V_w, np.ones(len(V_w), dtype=np.float64)]
                V_out = (T_inv @ V_h.T).T[:, :3]
            else:
                wxyt = _nav_world_xyt_from_obs(obs_pose)
                T_fix = _world_alignment_fixup_T(wxyt, T_base_w)
                V_out = (T_fix[:3, :3] @ V_w.T).T + T_fix[:3, 3]
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"
            seg = _safe_entity_segment(str(gname))
            rr.log(
                f"{entity_prefix}/{seg}",
                rr.Mesh3D(
                    vertex_positions=V_out.astype(np.float32),
                    triangle_indices=F.flatten(),
                ),
            )

