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

"""Helpers for :class:`RobosuiteZmqServer` model load: keyframe home pose and post-load diagnostics."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

if TYPE_CHECKING:
    from emet.robots.base import RobotSpec


def robosuite_post_load_debug_enabled(debug_molmospaces_spawn: bool) -> bool:
    """True when spawn debug is on or ``EMET_ROBOSUITE_POST_LOAD_DEBUG`` is set."""
    if debug_molmospaces_spawn:
        return True
    v = os.environ.get("EMET_ROBOSUITE_POST_LOAD_DEBUG", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def freejoint_qpos_qvel_addrs(model: mujoco.MjModel, base_body_name: str) -> tuple[int, int] | None:
    """Return ``(qposadr, dofadr)`` for the free joint on *base_body_name*, if any."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    if bid < 0:
        return None
    for j in range(model.njnt):
        if int(model.jnt_bodyid[j]) != bid:
            continue
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        return (int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]))
    return None


def _snapshot_freejoint_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> list[tuple[int, np.ndarray]]:
    """Copy 7-DoF ``qpos`` for every free joint (scene objects and optional free base)."""
    jnt_free = int(mujoco.mjtJoint.mjJNT_FREE)
    out: list[tuple[int, np.ndarray]] = []
    for j in range(int(model.njnt)):
        if int(model.jnt_type[j]) != jnt_free:
            continue
        qadr = int(model.jnt_qposadr[j])
        out.append((qadr, np.array(data.qpos[qadr : qadr + 7], dtype=np.float64, copy=True)))
    return out


def _restore_freejoint_qpos(data: mujoco.MjData, saved: list[tuple[int, np.ndarray]]) -> None:
    for qadr, q in saved:
        data.qpos[qadr : qadr + 7] = q


def _actuator_is_joint_velocity(model: mujoco.MjModel, aid: int) -> bool:
    """True for MuJoCo ``<velocity`` joint actuators (``gear[0] == 3``) or legacy ``wheel*`` names."""
    if abs(float(model.actuator_gear[int(aid), 0]) - 3.0) < 1e-6:
        return True
    nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
    return str(nm).startswith("wheel")


def snap_joint_qpos_to_ctrl_for_position_actuators(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    """Set hinge/slide ``qpos`` to ``ctrl`` for joint transmission actuators (skip velocity wheels).

    Merged MJCF keyframes compile a full ``key_qpos`` (scene + robot) while ``<key ctrl=.../>`` only
    lists robot controls. ``mj_resetDataKeyframe`` can then leave arm ``qpos`` near default zeros with
    ``ctrl`` already at home targets — the first ``mj_step`` calls slam joints toward limits.

    Returns:
        Number of scalar ``qpos`` entries written.
    """
    n_written = 0
    trn_joint = int(mujoco.mjtTrn.mjTRN_JOINT)
    for a in range(int(model.nu)):
        if int(model.actuator_trntype[a]) != trn_joint:
            continue
        jid = int(model.actuator_trnid[a, 0])
        if jid < 0:
            continue
        if _actuator_is_joint_velocity(model, a):
            continue
        jt = int(model.jnt_type[jid])
        if jt not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        qadr = int(model.jnt_qposadr[jid])
        data.qpos[qadr] = float(data.ctrl[a])
        n_written += 1
    return n_written


def robot_home_keyframe_name(spec: RobotSpec) -> str:
    """Per-robot MJCF keyframe id (avoids clashing with scene ``home`` on MolmoSpaces merges)."""
    return f"{spec.name}_home"


def resolve_robot_home_keyframe_id(model: mujoco.MjModel, spec: RobotSpec) -> int:
    """Return keyframe id for robot home pose, or -1 if missing."""
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, robot_home_keyframe_name(spec))
    if kid >= 0:
        return kid
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")


def apply_home_keyframe_preserving_base(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    spec: RobotSpec | None = None,
    key_name: str | None = None,
) -> bool:
    """If MJCF defines a robot home keyframe, reset to it while keeping free-joint poses.

    MolmoSpaces autoplace updates only the base; default compiled ``qpos`` can be a poor arm posture.
    ``mj_resetDataKeyframe`` applies the keyframe; we then restore every freejoint ``qpos`` from
    before the reset (robot base plus scene objects — merged table cubes sit in qpos 0–13 when
    the scene include comes first). On merged scenes, compiled ``key_qpos`` may disagree with
    ``key ctrl`` for the robot; we snap hinge/slide ``qpos`` to ``ctrl`` for matching actuators.

    Returns:
        True if the keyframe was applied, False if no home key or no base free joint.
    """
    addrs = freejoint_qpos_qvel_addrs(model, base_body_name)
    if addrs is None:
        return False
    if key_name is not None:
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    elif spec is not None:
        kid = resolve_robot_home_keyframe_id(model, spec)
    else:
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid < 0:
        return False
    vadr = int(addrs[1])
    saved_free = _snapshot_freejoint_qpos(model, data)
    mujoco.mj_resetDataKeyframe(model, data, kid)
    _restore_freejoint_qpos(data, saved_free)
    if vadr >= 0:
        data.qvel[vadr : vadr + 6] = 0.0
    data.qvel.fill(0.0)
    snap_joint_qpos_to_ctrl_for_position_actuators(model, data)
    _restore_freejoint_qpos(data, saved_free)
    mujoco.mj_forward(model, data)
    return True


def update_robot_qpos0_from_data(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec: RobotSpec,
) -> None:
    """Copy current ``qpos`` into ``qpos0`` for robot DOF only (scene objects keep compile-time ``qpos0``)."""
    addrs = freejoint_qpos_qvel_addrs(model, spec.base_link_name)
    if addrs is not None:
        qadr, _ = int(addrs[0]), int(addrs[1])
        model.qpos0[qadr : qadr + 7] = data.qpos[qadr : qadr + 7]
    planar_skip: set[str] = set()
    planar = getattr(spec, "planar_base_joint_names", None)
    if planar is not None and len(planar) == 3:
        planar_skip = {str(jname) for jname in planar}
    for jname in spec.joint_names:
        if str(jname) in planar_skip:
            continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            qadr = int(model.jnt_qposadr[jid])
            model.qpos0[qadr] = float(data.qpos[qadr])


def apply_home_keyframe_preserving_planar_base(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    planar_joint_names: tuple[str, str, str],
    base_body_name: str,
    spec: RobotSpec | None = None,
    key_name: str | None = None,
) -> bool:
    """Apply robot home keyframe while preserving planar base and scene freejoint ``qpos``.

    Merged table scenes put object ``<freejoint/>`` bodies before the robot, so a padded
    robot-only ``key_qpos`` would otherwise teleport cubes/cylinders to the origin.
    """
    if key_name is not None:
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    elif spec is not None:
        kid = resolve_robot_home_keyframe_id(model, spec)
    else:
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid < 0:
        return False
    saved: list[tuple[int, float]] = []
    for jname in planar_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(jname))
        if jid < 0:
            return False
        qadr = int(model.jnt_qposadr[jid])
        saved.append((qadr, float(data.qpos[qadr])))
    saved_free = _snapshot_freejoint_qpos(model, data)
    mujoco.mj_resetDataKeyframe(model, data, kid)
    _restore_freejoint_qpos(data, saved_free)
    for qadr, val in saved:
        data.qpos[qadr] = val
    data.qvel.fill(0.0)
    snap_joint_qpos_to_ctrl_for_position_actuators(model, data)
    # Velocity base actuators (gear≠3, e.g. xlerobot slide/yaw) still match mjTRN_JOINT; keep planar qpos.
    for qadr, val in saved:
        data.qpos[qadr] = val
    _restore_freejoint_qpos(data, saved_free)
    mujoco.mj_forward(model, data)
    return True


def apply_zero_joint_pose_preserving_base(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
) -> None:
    """Set hinge/slide ``qpos`` and all ``ctrl`` to zero; keep the base free joint pose if present."""
    addrs = freejoint_qpos_qvel_addrs(model, base_body_name)
    base_q: np.ndarray | None = None
    if addrs is not None:
        qadr, vadr = int(addrs[0]), int(addrs[1])
        base_q = np.array(data.qpos[qadr : qadr + 7], dtype=np.float64, copy=True)
    for jid in range(int(model.njnt)):
        jt = int(model.jnt_type[jid])
        if jt not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        qadr_j = int(model.jnt_qposadr[jid])
        data.qpos[qadr_j] = 0.0
    data.ctrl[:] = 0.0
    if base_q is not None:
        data.qpos[qadr : qadr + 7] = base_q
        if vadr >= 0:
            data.qvel[vadr : vadr + 6] = 0.0
    data.qvel.fill(0.0)
    mujoco.mj_forward(model, data)


def actuator_spec_vs_model_report(model: mujoco.MjModel, spec: RobotSpec) -> tuple[list[str], list[str]]:
    """Return (missing_in_model, extra_in_model_not_in_spec) actuator name lists (short names)."""
    spec_names = list(spec.actuator_names)
    missing: list[str] = []
    for aname in spec_names:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname) < 0:
            missing.append(aname)
    spec_set = set(spec_names)
    extra: list[str] = []
    for a in range(model.nu):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        if nm and nm not in spec_set:
            extra.append(str(nm))
    return missing, extra


def max_qvel_abs(data: mujoco.MjData) -> float:
    return float(np.max(np.abs(data.qvel))) if data.qvel.size else 0.0


def log_post_load_diagnostics(
    logger: Any,
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec: RobotSpec,
    stage: str,
    base_body_name: str,
) -> None:
    """Log timestep, actuator coverage, floor vs base height (debug only)."""
    missing, extra = actuator_spec_vs_model_report(model, spec)
    logger.info(
        f"[robosuite_load] stage={stage!r} timestep={float(model.opt.timestep):.5f} nu={model.nu} "
        f"nq={model.nq} nv={model.nv} robot={spec.name!r}"
    )
    if missing:
        logger.warning(f"[robosuite_load] spec actuators missing in MJCF: {missing}")
    if extra:
        tail = " …" if len(extra) > 20 else ""
        logger.info(f"[robosuite_load] MJCF actuators not in RobotSpec (informational): {extra[:20]}{tail}")

    try:
        from emet.simulation.molmospaces_spawn import resolve_floor_geom_name, walkable_floor_z_at_xy

        floor_nm = resolve_floor_geom_name(model) or "floor"
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        if bid >= 0:
            xy = (float(data.xpos[bid, 0]), float(data.xpos[bid, 1]))
            ray_excl = int(bid)
            floor = walkable_floor_z_at_xy(
                model, data, xy[0], xy[1], floor_geom_name=floor_nm, exclude_body_id=ray_excl
            )
            zb = float(data.xpos[bid, 2])
            logger.info(
                f"[robosuite_load] floor_geom={floor_nm!r} walkable_floor_z≈{floor} "
                f"base_body_z≈{zb:.3f} (under base XY)"
            )
    except Exception as e:
        logger.info(f"[robosuite_load] floor/base height diagnostic skipped ({e!r})")


def probe_max_qvel_unforced_steps(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    n_steps: int,
    sync_ctrl: Any,
    before_physics_step: Any | None = None,
) -> float | None:
    """Run *n_steps* ``mj_step`` with zero navigation after *sync_ctrl*; restore qpos/qvel/ctrl.

    *before_physics_step*: optional callable invoked with no args before each ``mj_step`` (e.g. apply
    stationary ``ctrl`` + spec hold in :class:`RobosuiteZmqServer`). Without it, ``ctrl`` is frozen
    at the post-*sync_ctrl* snapshot, which can mis-report stability for PD-driven models.

    Returns max |qvel| during probe, or None on failure. Caller should hold the sim lock.
    """
    if n_steps <= 0:
        return 0.0
    try:
        q0 = np.array(data.qpos, copy=True)
        v0 = np.array(data.qvel, copy=True)
        sync_ctrl()
        mujoco.mj_forward(model, data)
        # Snapshot ctrl *after* sync so restoring state puts back targets that match q0 (pre-probe).
        # Saving ctrl before sync and restoring it would re-apply stale/zero ctrl and break actuators.
        c0 = np.array(data.ctrl, copy=True) if data.ctrl.size else None
        mx = 0.0
        for _ in range(int(n_steps)):
            if before_physics_step is not None:
                before_physics_step()
            mujoco.mj_step(model, data)
            mx = max(mx, max_qvel_abs(data))
        np.copyto(data.qpos, q0)
        np.copyto(data.qvel, v0)
        if c0 is not None and data.ctrl.size:
            np.copyto(data.ctrl, c0)
        mujoco.mj_forward(model, data)
        return float(mx)
    except Exception:
        return None
