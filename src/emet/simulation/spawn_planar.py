# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import math
import os
from typing import Any

import mujoco
import numpy as np

import emet.utils.logger as log
from emet.robots.base import RobotSpec

logger = log.Logger(__name__)

from emet.simulation.spawn_debug import spawn_dbg
from emet.simulation.spawn_geom import (
    _MIN_UPWARD_CEILING_CLEARANCE_M,
    _bodies_descending_from,
    _clamp_xy_into_rect,
    _coarse_grid_xy_in_clip,
    _erode_xy_rect,
    collision_scene_xy_clip_rect,
    effective_floor_geom_name,
    horizontal_spawn_rejects_exterior_tongue,
    iter_annulus_xy_candidates,
    scene_collision_centroid_xy,
    upward_ray_hit_distance,
    walkable_floor_z_at_xy,
    worst_robot_nonfloor_contact_dist,
)
from emet.simulation.spawn_molmospaces import _ithor_occupancy_priority_xy


def planar_spawn_footprint_xy_margin_m(robot_spec: RobotSpec) -> float:
    """Footprint half-diagonal + pad for Robocasa walkable-clip erosion (planar and freejoint spawn)."""
    fp = robot_spec.footprint
    base_margin = float(
        0.5
        * math.hypot(
            float(fp.length) + abs(float(fp.length_offset)),
            float(fp.width) + abs(float(fp.width_offset)),
        )
        + 0.10
    )
    return base_margin + float(getattr(robot_spec, "planar_spawn_xy_extra_margin_m", 0.0) or 0.0)


def infer_planar_anchor_body_name(model: mujoco.MjModel, joint_names: tuple[str, str, str]) -> str | None:
    """Body that hosts the first planar slide joint (parent frame for slide axes)."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_names[0])
    if jid < 0:
        return None
    bid = int(model.jnt_bodyid[jid])
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    return str(name) if name else None


def world_se2_to_planar_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    anchor_body_name: str,
    world_x: float,
    world_y: float,
    world_yaw: float,
) -> tuple[float, float, float]:
    """Map desired ``base_link`` world *(x, y, yaw)* to planar *(slide_x, slide_y, hinge_yaw)* qpos.

    Slides move along the anchor body's local X/Y; after Robocasa merge, ``base_root`` is often
    translated and yawed, so **world** coordinates must not be written directly to the joints.
    """
    try:
        mujoco.mj_forward(model, data)
    except mujoco.FatalError as e:
        raise ValueError(f"mj_forward failed for planar anchor FK: {e}") from e
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, anchor_body_name)
    if bid < 0:
        raise ValueError(f"planar anchor body {anchor_body_name!r} not found")
    R = np.asarray(data.body(bid).xmat, dtype=np.float64).reshape(3, 3)
    pr = np.asarray(data.body(bid).xpos, dtype=np.float64)
    delta_xy = np.array([float(world_x) - pr[0], float(world_y) - pr[1]], dtype=np.float64)
    M = R[:2, :2]
    try:
        q_vec = np.linalg.solve(M, delta_xy)
    except np.linalg.LinAlgError:
        q_vec, *_ = np.linalg.lstsq(M, delta_xy, rcond=None)
    qx, qy = float(q_vec[0]), float(q_vec[1])

    cw = float(math.cos(float(world_yaw)))
    sw = float(math.sin(float(world_yaw)))
    rhs = np.array([cw, sw], dtype=np.float64)
    try:
        j_dir = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        j_dir, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    nj = float(np.linalg.norm(j_dir))
    if nj < 1e-9:
        j_yaw = 0.0
    else:
        j_dir = j_dir / nj
        j_yaw = float(math.atan2(float(j_dir[1]), float(j_dir[0])))
    return (qx, qy, j_yaw)


def write_planar_base_xyt(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    joint_names: tuple[str, str, str],
    world_x: float,
    world_y: float,
    world_yaw: float,
    anchor_body_name: str | None = None,
    base_body_name: str = "base_link",
) -> bool:
    """Set planar joints so *base_body_name* reaches world ``(world_x, world_y, world_yaw)`` after ``mj_forward``.

    Joint values are **not** raw world XY when the anchor body (parent of the first slide) is rotated
    or shifted (Robocasa merged robots). Pass ``anchor_body_name`` or rely on inference from
    ``joint_names[0]``.

    After dynamics (e.g. :meth:`RobosuiteZmqServer._stabilize_physics_state_after_load`), the kinematic
    subtree can sit in a state where a single analytic step does not match FK; we run a few Gauss–Newton
    style corrections in slide space, then yaw, until the reported *base* pose matches within tight
    tolerance.
    """
    for jn in joint_names:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn) < 0:
            return False
    anchor = anchor_body_name or infer_planar_anchor_body_name(model, joint_names)
    if anchor is None:
        return False
    try:
        qx, qy, j_yaw = world_se2_to_planar_qpos(
            model,
            data,
            anchor_body_name=anchor,
            world_x=float(world_x),
            world_y=float(world_y),
            world_yaw=float(world_yaw),
        )
    except ValueError:
        return False
    vals = (qx, qy, j_yaw)
    qadr_x = qadr_y = qadr_yaw = -1
    for jn, val in zip(joint_names, vals, strict=True):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        qadr = int(model.jnt_qposadr[jid])
        vadr = int(model.jnt_dofadr[jid])
        data.qpos[qadr] = float(val)
        if vadr >= 0:
            data.qvel[vadr] = 0.0
        if jn == joint_names[0]:
            qadr_x = qadr
        elif jn == joint_names[1]:
            qadr_y = qadr
        else:
            qadr_yaw = qadr
    for a in range(model.nu):
        jid_tr = int(model.actuator_trnid[a, 0])
        if jid_tr < 0:
            continue
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid_tr)
        if jname is not None and jname in joint_names:
            data.ctrl[a] = 0.0

    bid_bl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    bid_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, anchor)
    if bid_bl >= 0 and bid_a >= 0 and qadr_x >= 0 and qadr_y >= 0 and qadr_yaw >= 0:
        try:
            for _ in range(12):
                mujoco.mj_forward(model, data)
                ex = float(world_x) - float(data.body(bid_bl).xpos[0])
                ey = float(world_y) - float(data.body(bid_bl).xpos[1])
                if ex * ex + ey * ey < 2.25e-6:
                    break
                R = np.asarray(data.body(bid_a).xmat, dtype=np.float64).reshape(3, 3)
                M = R[:2, :2]
                delta = np.array([ex, ey], dtype=np.float64)
                try:
                    dq = np.linalg.solve(M, delta)
                except np.linalg.LinAlgError:
                    dq, *_ = np.linalg.lstsq(M, delta, rcond=None)
                data.qpos[qadr_x] = float(data.qpos[qadr_x]) + float(dq[0])
                data.qpos[qadr_y] = float(data.qpos[qadr_y]) + float(dq[1])
            for _ in range(8):
                mujoco.mj_forward(model, data)
                xmat = np.asarray(data.body(bid_bl).xmat, dtype=np.float64).reshape(3, 3)
                cur = float(math.atan2(xmat[1, 0], xmat[0, 0]))
                err = float(np.arctan2(np.sin(float(world_yaw) - cur), np.cos(float(world_yaw) - cur)))
                if abs(err) < 5e-4:
                    break
                data.qpos[qadr_yaw] = float(data.qpos[qadr_yaw]) + err
        except mujoco.FatalError:
            return False

    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        vadr = int(model.jnt_dofadr[jid])
        if vadr >= 0:
            data.qvel[vadr] = 0.0
    for a in range(model.nu):
        jid_tr = int(model.actuator_trnid[a, 0])
        if jid_tr < 0:
            continue
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid_tr)
        if jname is not None and jname in joint_names:
            data.ctrl[a] = 0.0
    try:
        mujoco.mj_forward(model, data)
    except mujoco.FatalError:
        return False
    return True


def try_xlerobot_molmospaces_planar_spawn(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    joint_names: tuple[str, str, str],
    merged_mjcf_path: str | None,
    environment: dict[str, Any] | None,
    max_points: int = 40,
) -> tuple[float, float, float] | None:
    """Fast MolmoSpaces spawn for XLeRobot: iTHOR occupancy XY + planar write (no full contact search).

    Full :func:`find_planar_base_xyt` is too slow / singular on dual-arm iTHOR merges; this moves the
    base onto a walkable floor cell so the robot is visible in the MuJoCo viewer and nav clients.
    """
    from emet.robots.xlerobot import apply_xlerobot_navigation_joint_pose

    apply_xlerobot_navigation_joint_pose(model, data)
    occ_xy, _ = _ithor_occupancy_priority_xy(
        merged_mjcf_path,
        environment,
        robot_root_body_name=base_body_name,
    )
    if not occ_xy:
        spawn_dbg("xlerobot_molmo_spawn: no iTHOR occupancy points")
        return None
    yaws = [float(k * math.pi / 4.0) for k in (0, 2, 4, 6)]
    for px, py in occ_xy[: max(1, int(max_points))]:
        for yaw in yaws:
            if write_planar_base_xyt(
                model,
                data,
                joint_names=joint_names,
                world_x=float(px),
                world_y=float(py),
                world_yaw=float(yaw),
                base_body_name=base_body_name,
            ):
                return (float(px), float(py), float(yaw))
    spawn_dbg("xlerobot_molmo_spawn: occupancy points did not map to planar joints")
    return None


def find_planar_base_xyt(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    joint_names: tuple[str, str, str],
    spawn_profile: str = "robocasa",
    floor_geom_name: str = "floor",
    scene_label: str | None = None,
    merged_mjcf_path: str | None = None,
    environment: dict[str, Any] | None = None,
    footprint_xy_margin_m: float = 0.35,
    spawn_hint_xyt: np.ndarray | None = None,
    anchor_body_name: str | None = None,
    clip_edge_pad_m: float = 0.22,
    clip_guard_body_names: tuple[str, ...] = (),
    clip_guard_pad_m: float = 0.18,
    robocasa_first_clearance_m: float | None = None,
) -> tuple[float, float, float] | None:
    """Search collision-free world `(x, y, yaw)` for planar slide+yaw bases (Robocasa + Maurice-style MJCF).

    Uses the same collision clip / floor probe / contact scoring ideas as free-joint Molmo spawn,
    but applies poses via :func:`write_planar_base_xyt`. For ``spawn_profile == \"robocasa\"`` we
    disable upward-ray rejection (open ceilings / cabinet tops) but **keep** the horizontal
    exterior-tongue filter so we do not pick the infinite floor outside the kitchen footprint.
    ``spawn_hint_xyt`` (world x, y, yaw from Robosuite ``init_robot_base_pos``) is tried first on
    ``spawn_profile == \"robocasa\"`` so innate_mars matches Stretch's kitchen placement instead of
    the walkable-clip centroid.

    Candidate *(x, y)* and *yaw* are **world** / kitchen-map coordinates; they are mapped to slide
    joint values using the planar anchor body (parent of the first slide), e.g. after Robocasa
    strip-and-replace when ``base_root`` carries a non-identity ``pos`` / ``quat``.

    ``clip_edge_pad_m`` insets the base link from the **un-eroded** scene walkable clip (room
    footprint projection).     ``clip_guard_body_names`` applies the same inset test for additional
    bodies (e.g. end-effector and wrist) when arm meshes are visual-only so
    :func:`worst_robot_nonfloor_contact_dist` does not see arm–counter penetration.
    ``robocasa_first_clearance_m`` (Robocasa profile) raises the first contact-clearance
    threshold before the ladder relaxes (helps mobile manipulators with visual-only arms).

    Returns world ``(x, y, yaw)`` matching :func:`get_base_xyt`-style conventions, or ``None``.
    """
    anchor_resolved = anchor_body_name or infer_planar_anchor_body_name(model, joint_names)
    if anchor_resolved is None:
        spawn_dbg("planar_find: could not infer anchor body for first planar joint; abort")
        return None

    floor_effective = effective_floor_geom_name(model, floor_geom_name)
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    robot_bodies: set[int] = set()
    if base_bid >= 0:
        robot_bodies = _bodies_descending_from(model, base_bid)
    ray_exclude = int(base_bid) if base_bid >= 0 else -1

    # Same exterior-geom filtering as Molmo: suppressing it inflates the clip with porch/deck
    # geoms and lets candidates land on open floor **outside** the kitchen footprint.
    xy_clip_scene = collision_scene_xy_clip_rect(
        model,
        data,
        robot_bodies=robot_bodies,
        floor_geom_name=floor_effective,
        margin=0.55,
        max_geom_rbound=15.0,
        suppress_exterior_filter=False,
    )
    inset = float(footprint_xy_margin_m) + 0.10
    xy_clip = _erode_xy_rect(xy_clip_scene, inset) if xy_clip_scene is not None else None
    if xy_clip is None:
        xy_clip = xy_clip_scene

    centroid = scene_collision_centroid_xy(
        model,
        data,
        robot_bodies=robot_bodies,
        floor_geom_name=floor_effective,
        max_geom_rbound=15.0,
        suppress_exterior_filter=False,
    )
    # Prefer annulus origin at **eroded clip center** so search stays inside walkable bounds
    # (scene centroid can sit near porch / merged-shell outliers).
    if xy_clip is not None:
        ox = 0.5 * float(xy_clip[0] + xy_clip[1])
        oy = 0.5 * float(xy_clip[2] + xy_clip[3])
    elif centroid is not None:
        ox, oy = float(centroid[0]), float(centroid[1])
    else:
        ox, oy = 0.0, 0.0
    if xy_clip is not None:
        ox, oy = _clamp_xy_into_rect(ox, oy, xy_clip)

    r_annulus_max = 3.5
    if xy_clip is not None:
        x0c, x1c, y0c, y1c = xy_clip
        w = float(x1c - x0c)
        h = float(y1c - y0c)
        half_diag = 0.5 * math.hypot(w, h)
        r_annulus_max = float(min(4.5, max(2.0, 0.38 * half_diag + 0.2)))

    base_candidates: list[tuple[float, float]] = list(
        iter_annulus_xy_candidates(
            r_max=r_annulus_max,
            xy_clip=xy_clip,
            xy_origin=(ox, oy),
            n_radii=22,
            base_angles_per_ring=14,
        )
    )
    if xy_clip is not None:
        seen = {(round(a, 2), round(b, 2)) for a, b in base_candidates}
        gstep = 0.48
        for px, py in _coarse_grid_xy_in_clip(xy_clip, step=gstep, max_points=520):
            key = (round(px, 2), round(py, 2))
            if key in seen:
                continue
            seen.add(key)
            base_candidates.append((float(px), float(py)))

    occ_xy, _ = _ithor_occupancy_priority_xy(
        merged_mjcf_path,
        environment,
        robot_root_body_name=base_body_name,
    )
    priority_xy: list[tuple[float, float]] = []
    hint_xy: tuple[float, float] | None = None
    hint_yaw: float | None = None
    if spawn_hint_xyt is not None:
        h = np.asarray(spawn_hint_xyt, dtype=np.float64).reshape(-1)
        if h.size >= 2:
            hint_xy = (float(h[0]), float(h[1]))
            priority_xy.append(hint_xy)
            if h.size >= 3:
                hint_yaw = float(h[2])
    if occ_xy:
        seen_xy = {(round(a, 2), round(b, 2)) for a, b in base_candidates}
        if hint_xy is not None:
            seen_xy.add((round(hint_xy[0], 2), round(hint_xy[1], 2)))
        for px, py in occ_xy:
            k = (round(px, 2), round(py, 2))
            if k in seen_xy:
                continue
            seen_xy.add(k)
            priority_xy.append((float(px), float(py)))

    candidates = priority_xy + base_candidates
    if hint_xy is not None:
        hx, hy = hint_xy
        candidates.sort(key=lambda p: (p[0] - hx) ** 2 + (p[1] - hy) ** 2)
    elif xy_clip is not None:
        cx = 0.5 * (xy_clip[0] + xy_clip[1])
        cy = 0.5 * (xy_clip[2] + xy_clip[3])
        candidates.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
    elif centroid is not None:
        cx, cy = float(centroid[0]), float(centroid[1])
        candidates.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)

    raw_xy_cap = os.environ.get("EMET_PLANAR_SPAWN_MAX_XY", "").strip()
    if raw_xy_cap.isdigit():
        max_xy_candidates = int(raw_xy_cap)
    elif spawn_profile == "molmospaces":
        # Dual-arm / wide-footprint robots (xlerobot) need tight caps: each (x,y,yaw) runs full contact probes.
        max_xy_candidates = 180
    elif spawn_profile == "robocasa":
        max_xy_candidates = 800
    else:
        max_xy_candidates = 0
    if max_xy_candidates > 0 and len(candidates) > max_xy_candidates:
        spawn_dbg(f"planar_find: capping n_xy {len(candidates)} -> {max_xy_candidates}")
        candidates = candidates[:max_xy_candidates]

    if spawn_profile == "molmospaces":
        yaws = [float(k * math.pi / 4.0) for k in (0, 2, 4, 6)]
    else:
        yaws = [float(k * math.pi / 4.0) for k in range(8)]
    if hint_yaw is not None:
        hy = float(hint_yaw)
        yaws = [hy] + [y for y in yaws if abs(float(np.arctan2(np.sin(y - hy), np.cos(y - hy)))) > 0.02]
    min_upward = -1.0 if spawn_profile == "robocasa" else float(_MIN_UPWARD_CEILING_CLEARANCE_M)
    clearance_passes: tuple[tuple[float, str], ...] = (
        (0.045, "clear045"),
        (0.028, "clear028"),
        (0.014, "clear014"),
        (0.005, "clear005"),
        (-5e-5, "clear_num"),
    )
    if spawn_profile == "molmospaces":
        clearance_passes = (
            (0.028, "clear028"),
            (0.014, "clear014"),
            (-5e-5, "clear_num"),
        )
    if spawn_profile == "robocasa" and robocasa_first_clearance_m is not None:
        fc = float(robocasa_first_clearance_m)
        clearance_passes = (
            (fc, "clear_robo_strict"),
            (max(0.028, 0.55 * fc), "clear_robo_mid"),
            (max(0.014, 0.32 * fc), "clear_robo_lo"),
            (0.005, "clear005"),
            (-5e-5, "clear_num"),
        )

    spawn_dbg(
        f"planar_find: profile={spawn_profile!r} scene={scene_label!r} n_xy={len(candidates)} "
        f"margin_m={footprint_xy_margin_m:.3f} clip={'yes' if xy_clip_scene is not None else 'no'}"
    )

    for min_clear, tag in clearance_passes:
        for x, y in candidates:
            for yaw in yaws:
                if not write_planar_base_xyt(
                    model,
                    data,
                    joint_names=joint_names,
                    world_x=float(x),
                    world_y=float(y),
                    world_yaw=float(yaw),
                    anchor_body_name=anchor_resolved,
                    base_body_name=base_body_name,
                ):
                    continue
                z_floor = walkable_floor_z_at_xy(
                    model, data, x, y, floor_geom_name=floor_effective, exclude_body_id=ray_exclude
                )
                if z_floor is None:
                    continue
                z_probe = float(z_floor) + 0.08
                if min_upward >= 0.0:
                    up_dist = upward_ray_hit_distance(model, data, x, y, z_probe, exclude_body_id=ray_exclude)
                    if up_dist is not None and up_dist < min_upward:
                        continue
                if horizontal_spawn_rejects_exterior_tongue(model, data, x, y, z_probe, exclude_body_id=ray_exclude):
                    continue
                worst = worst_robot_nonfloor_contact_dist(
                    model, data, base_body_name=base_body_name, floor_geom_name=floor_effective
                )
                if worst >= float(min_clear):
                    if xy_clip_scene is not None and base_bid >= 0:
                        bx = float(data.body(base_body_name).xpos[0])
                        by = float(data.body(base_body_name).xpos[1])
                        x0s, x1s, y0s, y1s = xy_clip_scene
                        pad = float(clip_edge_pad_m)
                        if not (x0s + pad <= bx <= x1s - pad and y0s + pad <= by <= y1s - pad):
                            continue
                        if clip_guard_body_names:
                            pg = float(clip_guard_pad_m)
                            skip_pose = False
                            for gnm in clip_guard_body_names:
                                gbid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(gnm))
                                if gbid < 0:
                                    skip_pose = True
                                    break
                                gx = float(data.body(gbid).xpos[0])
                                gy = float(data.body(gbid).xpos[1])
                                if not (x0s + pg <= gx <= x1s - pg and y0s + pg <= gy <= y1s - pg):
                                    skip_pose = True
                                    break
                            if skip_pose:
                                continue
                    spawn_dbg(f"planar_find: OK pass={tag!r} xy=({x:.3f},{y:.3f}) yaw={yaw:.3f} worst={worst:.5f}")
                    return (float(x), float(y), float(yaw))
    spawn_dbg(f"planar_find: failed scene={scene_label!r} profile={spawn_profile!r}")
    return None
