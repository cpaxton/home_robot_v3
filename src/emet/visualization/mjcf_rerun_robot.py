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

from emet.core.zmq_protocol import EMET_ZMQ_SESSION_KEY, read_emet_session
from emet.utils.geometry import nav_xyt_to_world_xyt

_MAX_BODIES = 72

_PLANAR_YAW_NAMES = frozenset({"base_yaw", "base_theta", "joint_mobile_base_theta"})


def _planar_base_joint_indices(joint_names: tuple[str, ...] | list[str]) -> tuple[int, int, int] | None:
    """Return indices of slide/slide/yaw planar base joints when present (innate_mars, stretch MJCF, …)."""
    if len(joint_names) < 3:
        return None
    n0, n1, n2 = str(joint_names[0]), str(joint_names[1]), str(joint_names[2])
    x_ok = n0 in ("base_x", "joint_mobile_base_x")
    y_ok = n1 in ("base_y", "joint_mobile_base_y")
    if x_ok and y_ok and n2 in _PLANAR_YAW_NAMES:
        return (0, 1, 2)
    return None


def _apply_planar_base_qpos_from_gps(
    model: Any,
    data: Any,
    obs_pose: dict[str, Any],
    joint_names: tuple[str, ...],
    planar_idx: tuple[int, int, int],
) -> None:
    """Episode-relative planar ``qpos`` for standalone MJCF (``world/robot`` supplies world compose)."""
    import mujoco

    gps = np.asarray(obs_pose.get("gps", np.zeros(2)), dtype=np.float64).reshape(-1)[:2]
    comp = np.asarray(obs_pose.get("compass", np.zeros(1)), dtype=np.float64).ravel()
    theta = float(comp[0]) if comp.size else 0.0
    vals = (float(gps[0]), float(gps[1]), theta)
    for vi, ji in enumerate(planar_idx):
        jname = joint_names[ji]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        qadr = int(model.jnt_qposadr[jid])
        data.qpos[qadr] = vals[vi]


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

    Free-joint and planar-base robots: base ``qpos`` from episode-relative ``gps``/``compass``
    only — **not** world ``joint[0:3]`` and not ``navigation_origin_xyt`` composed here — so
    ``world/robot`` (nav compose via :meth:`log_robot_xyt`) is not double-applied. Arm/head joints
    replay from ``joint``.
    """
    import mujoco

    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0

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
        z0 = float(model.qpos0[qadr + 2])
        qw, qx, qy, qz = _quat_wxyz_yaw(float(local_xyt[2]))
        data.qpos[qadr : qadr + 3] = [float(local_xyt[0]), float(local_xyt[1]), z0]
        data.qpos[qadr + 3 : qadr + 7] = [qw, qx, qy, qz]

    planar_idx = None if free_qadr is not None else _planar_base_joint_indices(joint_names)
    if planar_idx is not None:
        _apply_planar_base_qpos_from_gps(model, data, obs_pose, joint_names, planar_idx)

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

    skip_planar = set(planar_idx) if planar_idx is not None else set()
    for i, jname in enumerate(joint_names):
        if i in skip_planar:
            continue
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

    jhead = obs_pose.get("joint_head")
    if jhead is not None:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_head")
        if jid >= 0:
            qadr = int(model.jnt_qposadr[jid])
            data.qpos[qadr] = float(jhead)


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


def _Rz_mat(theta: float) -> np.ndarray:
    c, s = np.cos(float(theta)), np.sin(float(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _apply_base_yaw_fix_to_points(V: np.ndarray, yaw_rad: float) -> np.ndarray:
    if abs(float(yaw_rad)) < 1e-12:
        return V
    R = _Rz_mat(yaw_rad)
    return (R @ np.asarray(V, dtype=np.float64).T).T


def _mjcf_geom_rgba_u8(model: Any, gid: int, *, darken: float = 0.42) -> np.ndarray:
    rgba = np.asarray(model.geom_rgba[int(gid)], dtype=np.float64).reshape(4)
    if rgba[3] <= 1e-6:
        rgba = np.array([0.55, 0.55, 0.58, 1.0], dtype=np.float64)
    rgb = np.clip(np.round(rgba[:3] * 255.0 * float(darken)), 0, 255).astype(np.uint8)
    return rgb


# Semi-transparent, darkened MJCF mesh overlay in Rerun (0–1 albedo + alpha).
MJCF_RERUN_MESH_ALBEDO_FACTOR = (0.38, 0.38, 0.40, 0.52)


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
        self._hardware_model_patched = False
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

    def _maybe_patch_model_for_hardware_replay(self, obs_pose: dict[str, Any]) -> None:
        if self._hardware_model_patched:
            return
        from emet.robots.innate_mars.head_kinematics import (
            is_hardware_innate_mars_obs,
            patch_innate_mars_model_for_hardware_replay,
        )

        if is_hardware_innate_mars_obs(obs_pose):
            patch_innate_mars_model_for_hardware_replay(self.model)
            self._hardware_model_patched = True

    def _maybe_enrich_joint_head_from_camera(self, obs_pose: dict[str, Any]) -> dict[str, Any]:
        from emet.robots.innate_mars.head_kinematics import (
            enrich_obs_pose_joint_head_for_hardware_replay,
            is_hardware_innate_mars_obs,
        )

        if not is_hardware_innate_mars_obs(obs_pose):
            return obs_pose
        return enrich_obs_pose_joint_head_for_hardware_replay(self.model, obs_pose)

    def apply_and_log(self, obs_pose: dict[str, Any]) -> None:
        import mujoco
        import rerun as rr

        from emet.robots.innate_mars.head_kinematics import (
            HARDWARE_MJCF_VISUAL_YAW_RAD,
            is_hardware_innate_mars_obs,
            obs_pose_for_base_relative_mjcf_replay,
        )

        self._maybe_patch_model_for_hardware_replay(obs_pose)
        obs_pose = self._maybe_enrich_joint_head_from_camera(obs_pose)
        obs_pose = obs_pose_for_base_relative_mjcf_replay(obs_pose)
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
        hw_visual_yaw = HARDWARE_MJCF_VISUAL_YAW_RAD if is_hardware_innate_mars_obs(obs_pose) else 0.0
        R_fix = _Rz_mat(hw_visual_yaw)

        for bid, entity_path in zip(self._body_ids, self._body_paths, strict=True):
            T_w = _body_T_world(self.data, int(bid))
            T_rel = T_inv @ T_w
            if abs(hw_visual_yaw) > 1e-12:
                T_fix4 = np.eye(4, dtype=np.float64)
                T_fix4[:3, :3] = R_fix
                T_rel = T_fix4 @ T_rel
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
        self._hardware_model_patched = False
        self._free_qadr = _base_freejoint_qadr(self.model, self.base_link_name)
        self._nav_origin_slot: list[np.ndarray | None] = [None]
        self._geom_mesh_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._geom_color_cache: dict[int, np.ndarray] = {}
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
            self._geom_color_cache[gid] = _mjcf_geom_rgba_u8(self.model, gid)

    def _maybe_patch_model_for_hardware_replay(self, obs_pose: dict[str, Any]) -> None:
        if self._hardware_model_patched:
            return
        from emet.robots.innate_mars.head_kinematics import (
            is_hardware_innate_mars_obs,
            patch_innate_mars_model_for_hardware_replay,
        )

        if is_hardware_innate_mars_obs(obs_pose):
            patch_innate_mars_model_for_hardware_replay(self.model)
            self._hardware_model_patched = True

    def _maybe_enrich_joint_head_from_camera(self, obs_pose: dict[str, Any]) -> dict[str, Any]:
        from emet.robots.innate_mars.head_kinematics import (
            enrich_obs_pose_joint_head_for_hardware_replay,
            is_hardware_innate_mars_obs,
        )

        if not is_hardware_innate_mars_obs(obs_pose):
            return obs_pose
        return enrich_obs_pose_joint_head_for_hardware_replay(self.model, obs_pose)

    def sync_kinematics(
        self,
        obs_pose: dict[str, Any],
        *,
        zero_planar_base: bool = False,
    ) -> np.ndarray:
        """Apply *obs_pose* to the local MJCF and return ``base_link`` world ``(x, y, yaw)``."""
        import mujoco

        from emet.robots.innate_mars.head_kinematics import obs_pose_for_base_relative_mjcf_replay

        self._maybe_patch_model_for_hardware_replay(obs_pose)
        obs_pose = self._maybe_enrich_joint_head_from_camera(obs_pose)
        if zero_planar_base:
            obs_pose = obs_pose_for_base_relative_mjcf_replay(obs_pose)
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

        from emet.robots.innate_mars.head_kinematics import (
            HARDWARE_MJCF_VISUAL_YAW_RAD,
            is_hardware_innate_mars_obs,
        )

        use_base_relative = entity_prefix.startswith("world/robot")
        self.sync_kinematics(obs_pose, zero_planar_base=use_base_relative)
        root_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.base_link_name)
        T_base_w = _body_T_world(self.data, int(root_bid))
        T_inv = np.linalg.inv(T_base_w)
        hw_visual_yaw = (
            HARDWARE_MJCF_VISUAL_YAW_RAD if (use_base_relative and is_hardware_innate_mars_obs(obs_pose)) else 0.0
        )

        for gid, (V_loc, F) in self._geom_mesh_cache.items():
            R = np.asarray(self.data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
            p = np.asarray(self.data.geom_xpos[gid], dtype=np.float64).reshape(3)
            V_w = (R @ V_loc.T).T + p
            if use_base_relative:
                V_h = np.c_[V_w, np.ones(len(V_w), dtype=np.float64)]
                V_out = (T_inv @ V_h.T).T[:, :3]
                V_out = _apply_base_yaw_fix_to_points(V_out, hw_visual_yaw)
            else:
                wxyt = _nav_world_xyt_from_obs(obs_pose)
                T_fix = _world_alignment_fixup_T(wxyt, T_base_w)
                V_out = (T_fix[:3, :3] @ V_w.T).T + T_fix[:3, 3]
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"
            seg = _safe_entity_segment(str(gname))
            color = self._geom_color_cache.get(gid, np.array([140, 140, 145], dtype=np.uint8))
            vertex_colors = np.tile(color, (len(V_out), 1))
            rr.log(
                f"{entity_prefix}/{seg}",
                rr.Mesh3D(
                    vertex_positions=V_out.astype(np.float32),
                    triangle_indices=F.flatten(),
                    vertex_colors=vertex_colors,
                    albedo_factor=MJCF_RERUN_MESH_ALBEDO_FACTOR,
                ),
            )
