# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import mujoco
import numpy as np

import emet.utils.logger as log
from emet.simulation.molmospaces_spawn_metadata import load_molmospaces_spawn_metadata

logger = log.Logger(__name__)

from emet.simulation.spawn_geom import (
    _bodies_descending_from,
    _geom_body_is_robot,
    effective_floor_geom_name,
    walkable_floor_z_at_xy,
    worst_robot_nonfloor_contact_dist,
)


def _settle_foot_clearance_kw(target_foot_clearance_above_floor_m: float | None) -> dict[str, float]:
    """Keyword args for :func:`settle_free_base_z_to_floor` when JSON / caller overrides clearance."""
    if target_foot_clearance_above_floor_m is None:
        return {}
    return {"target_foot_clearance_above_floor_m": float(target_foot_clearance_above_floor_m)}


def _molmospaces_z_settle_options(
    model: mujoco.MjModel,
    *,
    base_body_name: str,
    robot_key: str | None,
) -> tuple[dict[str, float], set[int] | None]:
    """Foot-clearance kwargs and base+leg probe bodies for Molmo free-joint Z placement."""
    stf: float | None = None
    if robot_key:
        rk = str(robot_key).strip()
        if rk:
            meta = load_molmospaces_spawn_metadata(rk)
            if meta is not None and meta.molmospaces_target_foot_clearance_above_floor_m is not None:
                stf = float(meta.molmospaces_target_foot_clearance_above_floor_m)
    return (
        _settle_foot_clearance_kw(stf),
        support_collision_body_ids_for_base_z_placement(model, base_body_name),
    )


def write_freejoint_base_xyzw(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    x: float,
    y: float,
    z: float,
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> bool:
    """Set the 7-DOF free joint on *base_body_name* to position (x,y,z) and quaternion; zero base twist."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    if bid < 0:
        return False
    qw, qx, qy, qz = quat_wxyz
    for j in range(model.njnt):
        if int(model.jnt_bodyid[j]) != bid:
            continue
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        qadr = int(model.jnt_qposadr[j])
        vadr = int(model.jnt_dofadr[j])
        data.qpos[qadr : qadr + 3] = (x, y, z)
        data.qpos[qadr + 3 : qadr + 7] = (qw, qx, qy, qz)
        if vadr >= 0:
            data.qvel[vadr : vadr + 6] = 0.0
        return True
    return False


def restore_freejoint_base_from_model_qpos0(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
) -> bool:
    """Reset the base ``freejoint`` ``qpos``/``qvel`` from ``model.qpos0`` (compiled MJCF default).

    Spawn search moves the base to many candidate poses, often hoisting *z* very high between
    probes. If placement ultimately fails, callers should invoke this so the robot is not left
    invisible above the scene.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    if bid < 0:
        return False
    for j in range(model.njnt):
        if int(model.jnt_bodyid[j]) != bid:
            continue
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        qadr = int(model.jnt_qposadr[j])
        vadr = int(model.jnt_dofadr[j])
        data.qpos[qadr : qadr + 7] = model.qpos0[qadr : qadr + 7]
        if vadr >= 0:
            data.qvel[vadr : vadr + 6] = 0.0
        mujoco.mj_forward(model, data)
        return True
    return False


def _first_z_with_nonpenetrating_base(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    floor_geom_name: str,
    x: float,
    y: float,
    z_floor: float,
    z_min_above_floor: float,
    z_max_above_floor: float,
    n_z: int,
    min_clearance: float,
) -> float | None:
    """Sweep base height until robot–scene contacts (excluding floor) are non-penetrating."""
    for z in np.linspace(z_floor + z_min_above_floor, z_floor + z_max_above_floor, n_z):
        if not write_freejoint_base_xyzw(model, data, base_body_name=base_body_name, x=x, y=y, z=float(z)):
            break
        mujoco.mj_forward(model, data)
        worst = worst_robot_nonfloor_contact_dist(
            model, data, base_body_name=base_body_name, floor_geom_name=floor_geom_name
        )
        if worst >= min_clearance:
            return float(z)
    return None


def _min_robot_collision_geom_bottom_z(
    model: mujoco.MjModel, data: mujoco.MjData, robot_bodies: set[int]
) -> float | None:
    """Lowest world *z* of robot collision geoms (position minus ``geom_rbound``), or ``None``."""
    zmin = 1e30
    any_hit = False
    for g in range(model.ngeom):
        if not _geom_body_is_robot(model, g, robot_bodies):
            continue
        if not (int(model.geom_contype[g]) or int(model.geom_conaffinity[g])):
            continue
        if int(model.geom_type[g]) == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        p = data.geom_xpos[g]
        rb = float(model.geom_rbound[g])
        zmin = min(zmin, float(p[2]) - rb)
        any_hit = True
    if not any_hit:
        return None
    return float(zmin)


def support_collision_body_ids_for_base_z_placement(model: mujoco.MjModel, base_body_name: str) -> set[int] | None:
    """Body IDs for **base + swerve legs** only (exclude arms/torso for vertical *z* height probes."""
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    if root < 0:
        return None
    out: set[int] = {int(root)}
    for b in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if nm.startswith("steer_motor_link") or nm.startswith("wheel_motor_link"):
            out.add(int(b))
    return out if len(out) > 1 else None


def _torso_and_base_body_ids(model: mujoco.MjModel, base_body_name: str) -> set[int]:
    """``base_link`` plus ``torso_link*`` bodies (for visual hull placement)."""
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    if root < 0:
        return set()
    out: set[int] = {int(root)}
    for b in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if nm.startswith("torso_link"):
            out.add(int(b))
    return out


def _min_visual_geom_bottom_z(model: mujoco.MjModel, data: mujoco.MjData, body_ids: set[int]) -> float | None:
    """Lowest world *z* of **visual** (group 1) mesh geoms on *body_ids*, or ``None``."""
    zmin = 1e30
    any_hit = False
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) not in body_ids:
            continue
        if int(model.geom_group[g]) != 1:
            continue
        if int(model.geom_type[g]) == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        p = data.geom_xpos[g]
        rb = float(model.geom_rbound[g])
        zmin = min(zmin, float(p[2]) - rb)
        any_hit = True
    if not any_hit:
        return None
    return float(zmin)


def robot_placement_bottom_z(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    robot_bodies: set[int],
    zb_probe_bodies: set[int] | None = None,
) -> float | None:
    """Lowest *z* for vertical placement: collision on probe bodies plus torso/base visual hull."""
    col_src = robot_bodies if zb_probe_bodies is None else zb_probe_bodies
    z_col = _min_robot_collision_geom_bottom_z(model, data, col_src)
    z_vis = _min_visual_geom_bottom_z(model, data, _torso_and_base_body_ids(model, base_body_name))
    if z_col is None and z_vis is None:
        return None
    if z_col is None:
        return z_vis
    if z_vis is None:
        return z_col
    return float(min(z_col, z_vis))


def settle_free_base_z_to_floor(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    floor_geom_name: str,
    x: float,
    y: float,
    z_floor: float,
    z_start: float,
    robot_bodies: set[int],
    min_nonfloor_clearance: float,
    zb_probe_bodies: set[int] | None = None,
    max_steps: int = 320,
    step_m: float = 0.0028,
    target_foot_clearance_above_floor_m: float = 0.018,
    min_z_above_floor_m: float = -0.02,
) -> float | None:
    """Lower base *z* until collision hull is near *z_floor* while keeping non-floor contacts acceptable."""
    target_zb = float(z_floor) + float(target_foot_clearance_above_floor_m)
    z = float(z_start)
    last_good_z: float | None = None
    prev_z: float | None = None
    for _ in range(max_steps):
        if not write_freejoint_base_xyzw(model, data, base_body_name=base_body_name, x=x, y=y, z=z):
            break
        mujoco.mj_forward(model, data)
        worst = worst_robot_nonfloor_contact_dist(
            model, data, base_body_name=base_body_name, floor_geom_name=floor_geom_name
        )
        if worst < min_nonfloor_clearance - 1e-6:
            break
        zb = robot_placement_bottom_z(
            model,
            data,
            base_body_name=base_body_name,
            robot_bodies=robot_bodies,
            zb_probe_bodies=zb_probe_bodies,
        )
        if zb is None:
            break
        if zb <= target_zb + 2e-4:
            last_good_z = prev_z if prev_z is not None else z
            break
        prev_z = z
        last_good_z = z
        z -= step_m
        if z < z_floor + min_z_above_floor_m:
            break
    if last_good_z is None:
        return None
    write_freejoint_base_xyzw(model, data, base_body_name=base_body_name, x=x, y=y, z=float(last_good_z))
    mujoco.mj_forward(model, data)
    for _ in range(max_steps):
        zb = robot_placement_bottom_z(
            model,
            data,
            base_body_name=base_body_name,
            robot_bodies=robot_bodies,
            zb_probe_bodies=zb_probe_bodies,
        )
        if zb is None or zb >= target_zb - 1e-4:
            break
        z_try = float(last_good_z) + step_m
        if not write_freejoint_base_xyzw(model, data, base_body_name=base_body_name, x=x, y=y, z=z_try):
            break
        mujoco.mj_forward(model, data)
        worst = worst_robot_nonfloor_contact_dist(
            model, data, base_body_name=base_body_name, floor_geom_name=floor_geom_name
        )
        if worst < min_nonfloor_clearance - 1e-6:
            break
        last_good_z = z_try
    write_freejoint_base_xyzw(model, data, base_body_name=base_body_name, x=x, y=y, z=float(last_good_z))
    mujoco.mj_forward(model, data)
    return float(last_good_z)


def resettle_free_base_z_at_current_xy_preserving_yaw(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    floor_geom_name: str = "floor",
    robot_key: str | None = None,
    min_nonfloor_clearance: float = -5e-5,
    z_air_above_floor_m: float = 0.72,
) -> bool:
    """Re-run :func:`settle_free_base_z_to_floor` at the current base (x,y) with yaw preserved."""
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    if base_bid < 0:
        return False
    robot_bodies = _bodies_descending_from(model, base_bid)
    qadr: int | None = None
    vadr: int | None = None
    for j in range(model.njnt):
        if int(model.jnt_bodyid[j]) != base_bid:
            continue
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        qadr = int(model.jnt_qposadr[j])
        vadr = int(model.jnt_dofadr[j])
        break
    if qadr is None:
        return False

    backup = np.array(data.qpos[qadr : qadr + 7], dtype=np.float64, copy=True)
    x = float(data.qpos[qadr])
    y = float(data.qpos[qadr + 1])
    quat = tuple(float(v) for v in data.qpos[qadr + 3 : qadr + 7])

    floor_eff = effective_floor_geom_name(model, floor_geom_name)
    z_floor = walkable_floor_z_at_xy(model, data, x, y, floor_geom_name=floor_eff, exclude_body_id=int(base_bid))
    if z_floor is None:
        return False

    stf: float | None = None
    if robot_key:
        rk = str(robot_key).strip()
        if rk:
            meta = load_molmospaces_spawn_metadata(rk)
            if meta is not None and meta.molmospaces_target_foot_clearance_above_floor_m is not None:
                stf = float(meta.molmospaces_target_foot_clearance_above_floor_m)

    z_start = max(float(z_floor) + float(z_air_above_floor_m), float(backup[2]) + 0.42)
    if not write_freejoint_base_xyzw(model, data, base_body_name=base_body_name, x=x, y=y, z=z_start, quat_wxyz=quat):
        return False
    mujoco.mj_forward(model, data)

    zb_probe = support_collision_body_ids_for_base_z_placement(model, base_body_name)
    z_settled = settle_free_base_z_to_floor(
        model,
        data,
        base_body_name=base_body_name,
        floor_geom_name=floor_eff,
        x=x,
        y=y,
        z_floor=float(z_floor),
        z_start=z_start,
        robot_bodies=robot_bodies,
        min_nonfloor_clearance=min_nonfloor_clearance,
        zb_probe_bodies=zb_probe,
        **_settle_foot_clearance_kw(stf),
    )
    if z_settled is None:
        np.copyto(data.qpos[qadr : qadr + 7], backup)
        if vadr is not None and vadr >= 0:
            data.qvel[vadr : vadr + 6] = 0.0
        mujoco.mj_forward(model, data)
        return False
    if not _post_settle_pose_acceptable(
        model,
        data,
        base_body_name=base_body_name,
        floor_geom_name=floor_eff,
        z_floor=float(z_floor),
        robot_bodies=robot_bodies,
        min_nonfloor_clearance=min_nonfloor_clearance,
        zb_height_bodies=zb_probe,
    ):
        np.copyto(data.qpos[qadr : qadr + 7], backup)
        if vadr is not None and vadr >= 0:
            data.qvel[vadr : vadr + 6] = 0.0
        mujoco.mj_forward(model, data)
        return False
    return True


def _post_settle_pose_acceptable(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    floor_geom_name: str,
    z_floor: float,
    robot_bodies: set[int],
    min_nonfloor_clearance: float,
    zb_height_bodies: set[int] | None = None,
    max_foot_above_floor_m: float = 0.14,
    max_foot_below_floor_m: float = 0.05,
) -> bool:
    """After :func:`settle_free_base_z_to_floor`, reject poses that still clip scene or float far above floor."""
    mujoco.mj_forward(model, data)
    worst = worst_robot_nonfloor_contact_dist(
        model, data, base_body_name=base_body_name, floor_geom_name=floor_geom_name
    )
    if worst < min_nonfloor_clearance - 1e-6:
        return False
    zb = robot_placement_bottom_z(
        model,
        data,
        base_body_name=base_body_name,
        robot_bodies=robot_bodies,
        zb_probe_bodies=zb_height_bodies,
    )
    if zb is None:
        return False
    if zb < z_floor - max_foot_below_floor_m:
        return False
    if zb > z_floor + max_foot_above_floor_m:
        return False
    return True
