# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import math
import os
from typing import Any

import cv2
import mujoco
import numpy as np

import emet.utils.logger as log

logger = log.Logger(__name__)

from emet.simulation.spawn_debug import _spawn_debug_emit_topdown, spawn_dbg, spawn_debug_enabled
from emet.simulation.spawn_geom import (
    _DEFAULT_MOLMOSPACES_GRID_IN_CLIP_MAX,
    _DEFAULT_MOLMOSPACES_GRID_STEP_M,
    _MIN_UPWARD_CEILING_CLEARANCE_M,
    _bodies_descending_from,
    _clamp_xy_into_rect,
    _coarse_grid_xy_in_clip,
    _erode_xy_rect,
    _max_xy_distance_to_rect_corners,
    collision_scene_xy_clip_rect,
    effective_floor_geom_name,
    horizontal_spawn_rejects_exterior_tongue,
    iter_annulus_xy_candidates,
    scene_collision_centroid_xy,
    upward_ray_hit_distance,
    walkable_floor_z_at_xy,
    worst_robot_nonfloor_contact_dist,
)
from emet.simulation.spawn_settle import (
    _first_z_with_nonpenetrating_base,
    _molmospaces_z_settle_options,
    _post_settle_pose_acceptable,
    restore_freejoint_base_from_model_qpos0,
    settle_free_base_z_to_floor,
    write_freejoint_base_xyzw,
)


def want_molmospaces_autoplace(
    *,
    environment: dict[str, Any] | None,
    scene_source_basename: str | None,
    molmospaces_autoplace_env: str | None = None,
) -> bool:
    """Whether to run MolmoSpaces-style free-base placement after MJCF load.

    ``EMET_MOLMOSPACES_AUTOPLACE`` (default ``1``): ``0``/``false``/``no``/``off`` disables.
    ``extended``/``2`` also enables heuristics for common renamed merges (e.g. *FloorPlan* / *ithor*
    in the basename when the env value is ``extended`` or ``2``).
    """
    raw = (
        molmospaces_autoplace_env
        if molmospaces_autoplace_env is not None
        else os.environ.get("EMET_MOLMOSPACES_AUTOPLACE", "1")
    )
    v = str(raw).strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    env = environment
    if env and env.get("kind") == "molmospaces":
        return True
    bn = (scene_source_basename or "").lower()
    if bn.startswith("molmospaces_merged"):
        return True
    if "molmospaces" in bn or "mlspaces" in bn:
        return True
    if "merged" in bn and bn.endswith(".xml"):
        return True
    if v in ("extended", "2"):
        if bn.endswith(".xml") and ("floorplan" in bn or "ithor" in bn):
            return True
    return False


def _ithor_free_xy_order_by_interior_depth(ithor_map: Any, fp: np.ndarray) -> np.ndarray:
    """Row indices into *fp* sorting free samples by descending interior depth (grid pixels).

    OpenCV ``distanceTransform`` distance at a free cell is the distance in pixels to the nearest
    blocked cell, so orthographic **deep interior** points sort first for spawn priority.

    ``get_free_points()`` returns either ``(N, 2)`` or ``(N, 3)`` world coordinates; both are
    accepted for ``pos_m_to_px`` (which requires a homogeneous *xy* with a *z* column).
    """
    occ = np.asarray(ithor_map.occupancy, dtype=np.uint8)
    H, W = occ.shape
    src = occ.copy()
    dt = cv2.distanceTransform(src, cv2.DIST_L2, 5)
    fp64 = np.asarray(fp, dtype=np.float64)
    if fp64.ndim != 2 or fp64.shape[1] < 2:
        return np.zeros(0, dtype=np.int64)
    n = int(fp64.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if fp64.shape[1] == 2:
        xyz = np.concatenate([fp64, np.zeros((n, 1), dtype=np.float64)], axis=1)
    else:
        xyz = fp64[:, :3].copy()
    rc = np.asarray(ithor_map.pos_m_to_px(xyz), dtype=np.int64).reshape(-1, 2)
    rows = np.clip(rc[:, 0], 0, H - 1)
    cols = np.clip(rc[:, 1], 0, W - 1)
    depths = dt[rows, cols]
    return np.argsort(-depths, kind="stable")


def _ithor_occupancy_priority_xy(
    merged_mjcf_path: str | None,
    environment: dict[str, Any] | None,
    *,
    robot_root_body_name: str | None = None,
    agent_radius: float = 0.32,
    px_per_m: int = 120,
    max_points: int = 900,
) -> tuple[list[tuple[float, float]], Any | None]:
    """Molmo-style orthographic occupancy free-space samples (vendored iTHORMap), optional XY priority.

    Returns ``(priority_xy_samples, ithor_map_or_none)``. The map reference is kept **only** when
    :func:`spawn_debug_enabled` so we can log a top-down ASCII / optional PNG without loading MJCF twice.

    Free cells are ordered by **interior depth** (distance in the occupancy grid to the nearest
    blocked cell), not a random subsample, so spawn tries room interiors before exterior tongues.
    ``EMET_MOLMOSPACES_OCC_PRIORITY_MAX`` overrides *max_points* (clamped to ``[200, 12000]``).
    """
    raw = os.environ.get("EMET_MOLMOSPACES_OCC_MAP", "1").strip().lower()
    if raw in ("0", "false", "no", "off") or not merged_mjcf_path:
        return [], None
    env = environment or {}
    scene = str(env.get("scene", "")).lower()
    path_l = merged_mjcf_path.lower()
    if "ithor" not in scene and "ithor" not in path_l:
        return [], None
    raw_cap = os.environ.get("EMET_MOLMOSPACES_OCC_PRIORITY_MAX", "").strip()
    eff_cap = int(raw_cap) if raw_cap.isdigit() else int(max_points)
    eff_cap = max(200, min(eff_cap, 12_000))
    try:
        from emet.simulation.molmo_occupancy.ithor_map import iTHORMap

        th = iTHORMap.from_mj_model_path(
            merged_mjcf_path,
            camera=None,
            agent_radius=agent_radius,
            px_per_m=px_per_m,
            robot_root_body_name=robot_root_body_name or "base_link",
        )
        fp = th.get_free_points()
    except Exception as e:
        logger.warning(f"MolmoSpaces occupancy map skipped ({e!r}).")
        return [], None
    if fp.size == 0:
        return [], None
    n = min(eff_cap, len(fp))
    seed = int(os.environ.get("EMET_MOLMOSPACES_OCC_SEED", "0") or 0)
    rng = np.random.default_rng(seed)
    order_mode = "interior_dt"
    try:
        order = _ithor_free_xy_order_by_interior_depth(th, fp)
    except Exception as e:
        order_mode = f"shuffle_fallback({e!r})"
        logger.warning(f"MolmoSpaces occupancy interior-depth sort skipped ({e!r}); using shuffle.")
        order = rng.permutation(len(fp))
    idx = order[:n]
    out = [(float(fp[i, 0]), float(fp[i, 1])) for i in idx]
    spawn_dbg(
        f"occupancy_map: n_free={len(fp)} priority_sample={len(out)} order={order_mode} "
        f"path={merged_mjcf_path!r} agent_r={agent_radius} px_per_m={px_per_m}"
    )
    th_dbg: Any | None = th if spawn_debug_enabled() else None
    return out, th_dbg


def _world_xy_to_occ_cell(ithor_map: Any, x: float, y: float) -> tuple[int, int] | None:
    """Map world *x*, *y* to occupancy array indices ``(row, col)``, or ``None`` if off-map."""
    raw = ithor_map.pos_m_to_px(np.array([[float(x), float(y), 0.0]], dtype=np.float64))
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        return None
    r, c = int(round(float(arr[0]))), int(round(float(arr[1])))
    H, W = int(ithor_map.occupancy.shape[0]), int(ithor_map.occupancy.shape[1])
    if r < 0 or c < 0 or r >= H or c >= W:
        return None
    return r, c


def _find_molmospaces_freejoint_xyz_pass(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    floor_effective: str,
    z_margins: tuple[float, ...],
    min_nonfloor_clearance: float,
    min_upward_clearance: float,
    max_geom_rbound: float,
    clip_erode_m: float,
    suppress_exterior_filter: bool,
    xy_priority: list[tuple[float, float]] | None = None,
    settle_kw: dict[str, float] | None = None,
    zb_probe_bodies: set[int] | None = None,
) -> tuple[float, float, float] | None:
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    robot_bodies: set[int] = set()
    if base_bid >= 0:
        robot_bodies = _bodies_descending_from(model, base_bid)
    ray_exclude = int(base_bid) if base_bid >= 0 else -1
    xy_clip_raw = collision_scene_xy_clip_rect(
        model,
        data,
        robot_bodies=robot_bodies,
        floor_geom_name=floor_effective,
        max_geom_rbound=max_geom_rbound,
        suppress_exterior_filter=suppress_exterior_filter,
    )
    xy_clip = _erode_xy_rect(xy_clip_raw, clip_erode_m) if xy_clip_raw is not None else None
    if xy_clip is None:
        xy_clip = xy_clip_raw

    centroid = scene_collision_centroid_xy(
        model,
        data,
        robot_bodies=robot_bodies,
        floor_geom_name=floor_effective,
        max_geom_rbound=max_geom_rbound,
        suppress_exterior_filter=suppress_exterior_filter,
    )
    ox, oy = 0.0, 0.0
    if centroid is not None:
        ox, oy = float(centroid[0]), float(centroid[1])
    elif xy_clip is not None:
        ox = 0.5 * float(xy_clip[0] + xy_clip[1])
        oy = 0.5 * float(xy_clip[2] + xy_clip[3])
    if xy_clip is not None:
        # Collision centroid can sit outside the *eroded* clip (e.g. porch geoms below the room
        # AABB). Annulus + clip filter then misses most walkable floor; clamp into the search rect.
        ox, oy = _clamp_xy_into_rect(ox, oy, xy_clip)

    r_annulus_max = 3.2
    if xy_clip is not None:
        x0c, x1c, y0c, y1c = xy_clip
        half_diag = 0.5 * math.hypot(x1c - x0c, y1c - y0c)
        # Must reach every corner from (ox,oy) plus inner ring (r_min ~0.35).
        reach_corners = _max_xy_distance_to_rect_corners(ox, oy, xy_clip) + 0.42
        r_annulus_max = float(min(14.5, max(3.8, reach_corners, half_diag + 0.9)))

    _nr = os.environ.get("EMET_MOLMOSPACES_ANNULUS_N_RADII", "").strip()
    _na = os.environ.get("EMET_MOLMOSPACES_ANNULUS_BASE_ANGLES", "").strip()
    n_radii_ann = int(_nr) if _nr.isdigit() else 28
    n_radii_ann = max(14, min(n_radii_ann, 60))
    base_ang_ann = int(_na) if _na.isdigit() else 15
    base_ang_ann = max(10, min(base_ang_ann, 48))

    base_candidates = list(
        iter_annulus_xy_candidates(
            r_max=r_annulus_max,
            xy_clip=xy_clip,
            xy_origin=(ox, oy),
            n_radii=n_radii_ann,
            base_angles_per_ring=base_ang_ann,
        )
    )
    if xy_clip is not None:
        seen = {(round(a, 2), round(b, 2)) for a, b in base_candidates}
        _gs = os.environ.get("EMET_MOLMOSPACES_GRID_STEP_M", "").strip()
        _gm = os.environ.get("EMET_MOLMOSPACES_GRID_MAX_POINTS", "").strip()
        gstep = float(_gs) if _gs else float(_DEFAULT_MOLMOSPACES_GRID_STEP_M)
        gstep = max(0.28, min(gstep, 1.2))
        gmax = int(_gm) if _gm.isdigit() else int(_DEFAULT_MOLMOSPACES_GRID_IN_CLIP_MAX)
        gmax = max(120, min(gmax, 2000))
        for px, py in _coarse_grid_xy_in_clip(xy_clip, step=gstep, max_points=gmax):
            key = (round(px, 2), round(py, 2))
            if key in seen:
                continue
            seen.add(key)
            base_candidates.append((px, py))

    if xy_clip is not None:
        cx = 0.5 * (xy_clip[0] + xy_clip[1])
        cy = 0.5 * (xy_clip[2] + xy_clip[3])
        base_candidates.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
    elif centroid is not None:
        cx, cy = float(centroid[0]), float(centroid[1])
        base_candidates.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)

    # iTHOR occupancy samples must be tried *first* in this order. Previously they were prepended
    # then the combined list was sorted by distance to clip center, burying them behind hundreds
    # of annulus/grid points and letting stale robot poses dominate early failures.
    priority_xy: list[tuple[float, float]] = []
    if xy_priority:
        seen_xy = {(round(a, 2), round(b, 2)) for a, b in base_candidates}
        for px, py in xy_priority:
            k = (round(px, 2), round(py, 2))
            if k in seen_xy:
                continue
            seen_xy.add(k)
            priority_xy.append((float(px), float(py)))
    candidates = priority_xy + base_candidates

    spawn_dbg(
        f"pass: origin=({ox:.3f},{oy:.3f}) r_annulus_max={r_annulus_max:.3f} n_candidates={len(candidates)} "
        f"n_occ_priority={len(priority_xy)} "
        f"clip_erode={clip_erode_m} max_rbound={max_geom_rbound} suppress_exterior={suppress_exterior_filter}"
    )

    _settle_kw = dict(settle_kw or {})
    best_xy: tuple[float, float, float, float] | None = None  # x, y, z_floor, worst_score
    z_air_probe = max(8.0, 3.0 * float(model.stat.extent))
    for x, y in candidates:
        # Clear bad penetration from the previous (x,y) attempt before floor / ceiling rays.
        if not write_freejoint_base_xyzw(
            model, data, base_body_name=base_body_name, x=float(x), y=float(y), z=z_air_probe
        ):
            continue
        mujoco.mj_forward(model, data)
        z_floor = walkable_floor_z_at_xy(
            model, data, x, y, floor_geom_name=floor_effective, exclude_body_id=ray_exclude
        )
        if z_floor is None:
            continue
        z_probe = float(z_floor) + 0.08
        up_dist = upward_ray_hit_distance(model, data, x, y, z_probe, exclude_body_id=ray_exclude)
        # Open sky (no upward hit) is common in iTHOR MJCF without an explicit ceiling geom; do not
        # treat that like "void under a shelf". Reject only a *close* ceiling when a hit exists.
        if up_dist is not None and up_dist < min_upward_clearance:
            continue
        if horizontal_spawn_rejects_exterior_tongue(model, data, x, y, z_probe, exclude_body_id=ray_exclude):
            continue
        for zm in z_margins:
            z = z_floor + float(zm)
            if not write_freejoint_base_xyzw(model, data, base_body_name=base_body_name, x=x, y=y, z=z):
                break
            mujoco.mj_forward(model, data)
            worst = worst_robot_nonfloor_contact_dist(
                model, data, base_body_name=base_body_name, floor_geom_name=floor_effective
            )
            if worst >= min_nonfloor_clearance:
                z_settled = settle_free_base_z_to_floor(
                    model,
                    data,
                    base_body_name=base_body_name,
                    floor_geom_name=floor_effective,
                    x=x,
                    y=y,
                    z_floor=float(z_floor),
                    z_start=z,
                    robot_bodies=robot_bodies,
                    min_nonfloor_clearance=min_nonfloor_clearance,
                    zb_probe_bodies=zb_probe_bodies,
                    **_settle_kw,
                )
                if z_settled is None:
                    continue
                if not _post_settle_pose_acceptable(
                    model,
                    data,
                    base_body_name=base_body_name,
                    floor_geom_name=floor_effective,
                    z_floor=float(z_floor),
                    robot_bodies=robot_bodies,
                    min_nonfloor_clearance=min_nonfloor_clearance,
                    zb_height_bodies=zb_probe_bodies,
                ):
                    continue
                return (x, y, z_settled)
            if best_xy is None or worst > best_xy[3]:
                best_xy = (x, y, float(z_floor), worst)

    if best_xy is None:
        return None
    bx, by, bz_floor, _ = best_xy
    z_ref = _first_z_with_nonpenetrating_base(
        model,
        data,
        base_body_name=base_body_name,
        floor_geom_name=floor_effective,
        x=bx,
        y=by,
        z_floor=bz_floor,
        z_min_above_floor=0.06,
        z_max_above_floor=1.15,
        n_z=48,
        min_clearance=min_nonfloor_clearance,
    )
    if z_ref is not None:
        z_settled = settle_free_base_z_to_floor(
            model,
            data,
            base_body_name=base_body_name,
            floor_geom_name=floor_effective,
            x=bx,
            y=by,
            z_floor=bz_floor,
            z_start=z_ref,
            robot_bodies=robot_bodies,
            min_nonfloor_clearance=min_nonfloor_clearance,
            zb_probe_bodies=zb_probe_bodies,
            **_settle_kw,
        )
        if z_settled is None:
            return None
        if not _post_settle_pose_acceptable(
            model,
            data,
            base_body_name=base_body_name,
            floor_geom_name=floor_effective,
            z_floor=float(bz_floor),
            robot_bodies=robot_bodies,
            min_nonfloor_clearance=min_nonfloor_clearance,
            zb_height_bodies=zb_probe_bodies,
        ):
            return None
        return (bx, by, z_settled)
    return None


def _try_spawn_at_xy_candidates(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    floor_effective: str,
    robot_bodies: set[int],
    ray_exclude: int,
    x: float,
    y: float,
    z_margins: tuple[float, ...],
    min_nonfloor_clearance: float,
    min_upward_clearance: float,
    settle_kw: dict[str, float] | None = None,
    zb_probe_bodies: set[int] | None = None,
) -> tuple[float, float, float] | None:
    """Single (x,y): walkable floor, upward clearance, z margin sweep + settle (shared by fallbacks)."""
    _settle_kw = dict(settle_kw or {})
    z_air = max(8.0, 3.0 * float(model.stat.extent))
    if not write_freejoint_base_xyzw(model, data, base_body_name=base_body_name, x=float(x), y=float(y), z=z_air):
        return None
    mujoco.mj_forward(model, data)
    z_floor = walkable_floor_z_at_xy(model, data, x, y, floor_geom_name=floor_effective, exclude_body_id=ray_exclude)
    if z_floor is None:
        return None
    z_probe = float(z_floor) + 0.08
    up_dist = upward_ray_hit_distance(model, data, x, y, z_probe, exclude_body_id=ray_exclude)
    if up_dist is not None and up_dist < min_upward_clearance:
        return None
    if horizontal_spawn_rejects_exterior_tongue(model, data, x, y, z_probe, exclude_body_id=ray_exclude):
        return None
    for zm in z_margins:
        z = z_floor + float(zm)
        if not write_freejoint_base_xyzw(model, data, base_body_name=base_body_name, x=x, y=y, z=z):
            break
        mujoco.mj_forward(model, data)
        worst = worst_robot_nonfloor_contact_dist(
            model, data, base_body_name=base_body_name, floor_geom_name=floor_effective
        )
        if worst >= min_nonfloor_clearance:
            z_settled = settle_free_base_z_to_floor(
                model,
                data,
                base_body_name=base_body_name,
                floor_geom_name=floor_effective,
                x=x,
                y=y,
                z_floor=float(z_floor),
                z_start=z,
                robot_bodies=robot_bodies,
                min_nonfloor_clearance=min_nonfloor_clearance,
                zb_probe_bodies=zb_probe_bodies,
                **_settle_kw,
            )
            if z_settled is None:
                continue
            if not _post_settle_pose_acceptable(
                model,
                data,
                base_body_name=base_body_name,
                floor_geom_name=floor_effective,
                z_floor=float(z_floor),
                robot_bodies=robot_bodies,
                min_nonfloor_clearance=min_nonfloor_clearance,
                zb_height_bodies=zb_probe_bodies,
            ):
                continue
            return (x, y, z_settled)
    return None


def _fallback_spawn_near_clip_center(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    floor_effective: str,
    robot_bodies: set[int],
    ray_exclude: int,
    xy_clip_raw: tuple[float, float, float, float] | None,
    z_margins: tuple[float, ...],
    min_nonfloor_clearance: float,
    settle_kw: dict[str, float] | None = None,
    zb_probe_bodies: set[int] | None = None,
) -> tuple[float, float, float] | None:
    """Fallback: a few XY points near the collision clip center with looser ceiling / contact."""
    if xy_clip_raw is None:
        return None
    x0, x1, y0, y1 = xy_clip_raw
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    w = x1 - x0
    h = y1 - y0
    dxy = min(0.55, 0.12 * min(w, h))
    offsets = (
        (0.0, 0.0),
        (dxy, 0.0),
        (-dxy, 0.0),
        (0.0, dxy),
        (0.0, -dxy),
        (dxy, dxy),
        (-dxy, dxy),
        (dxy, -dxy),
        (-dxy, -dxy),
    )
    for ox, oy in offsets:
        px, py = cx + ox, cy + oy
        if not (x0 <= px <= x1 and y0 <= py <= y1):
            continue
        for min_up in (0.12, 0.06, 0.035):
            for clear in (min_nonfloor_clearance, -0.002, -0.006):
                got = _try_spawn_at_xy_candidates(
                    model,
                    data,
                    base_body_name=base_body_name,
                    floor_effective=floor_effective,
                    robot_bodies=robot_bodies,
                    ray_exclude=ray_exclude,
                    x=px,
                    y=py,
                    z_margins=z_margins,
                    min_nonfloor_clearance=float(clear),
                    min_upward_clearance=min_up,
                    settle_kw=settle_kw,
                    zb_probe_bodies=zb_probe_bodies,
                )
                if got is not None:
                    return got
    return None


def find_molmospaces_freejoint_xyz(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    floor_geom_name: str = "floor",
    z_margins: tuple[float, ...] = (
        0.08,
        0.10,
        0.12,
        0.14,
        0.18,
        0.22,
        0.26,
        0.30,
        0.34,
        0.38,
        0.42,
        0.48,
        0.54,
    ),
    min_nonfloor_clearance: float = -5e-5,
    scene_label: str | None = None,
    merged_mjcf_path: str | None = None,
    environment: dict[str, Any] | None = None,
    robot_key: str | None = None,
) -> tuple[float, float, float] | None:
    """Return a world-space base position (x, y, z) for the robot free joint, or None to keep defaults.

    *robot_key* (e.g. ``RobotSpec.name``) loads optional ``molmospaces_spawn.json`` foot clearance
    and uses base+leg collision bodies for vertical placement (rby1 / galaxea_r1).

    Requires **no meaningful penetration** (``contact.dist >= min_nonfloor_clearance``) vs scene
    geoms other than *floor* (tiny negative values are treated as numerical noise).
    Uses a coarse polar XY search (clipped to scene collision footprint when available), requires
    a resolvable floor under (x,y), rejects **too-close** upward ceiling hits when an upward ray
    exists (open sky with no ceiling geom is allowed for iTHOR-style merges), rejects **exterior
    floor tongues** via horizontal cardinal ``mj_ray`` heuristics, then a finer vertical sweep on
    the best XY when needed.
    Returns ``None`` if no valid pose is found (caller must not apply a bogus exterior placement).
    On failure, resets the base free joint from ``model.qpos0`` so ``data`` is not left at a probe
    pose (e.g. very high *z*). On success, leaves *data* at the chosen pose with ``mj_forward``.
    """
    floor_effective = effective_floor_geom_name(model, floor_geom_name)
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    robot_bodies: set[int] = set()
    if base_bid >= 0:
        robot_bodies = _bodies_descending_from(model, base_bid)

    initial_base_xy: tuple[float, float] | None = None
    if base_bid >= 0:
        pos = data.xpos[base_bid]
        initial_base_xy = (float(pos[0]), float(pos[1]))

    clip_probe = collision_scene_xy_clip_rect(
        model, data, robot_bodies=robot_bodies, floor_geom_name=floor_effective, max_geom_rbound=15.0
    )
    if clip_probe is not None:
        spawn_dbg(
            f"find: clip_probe=({clip_probe[0]:.3f},{clip_probe[1]:.3f})x({clip_probe[2]:.3f},{clip_probe[3]:.3f}) "
            f"floor_effective={floor_effective!r} label={scene_label!r}"
        )
    else:
        spawn_dbg(f"find: clip_probe=None floor_effective={floor_effective!r} label={scene_label!r}")

    occ_xy, occ_map_dbg = _ithor_occupancy_priority_xy(
        merged_mjcf_path, environment, robot_root_body_name=base_body_name
    )
    settle_kw, zb_probe = _molmospaces_z_settle_options(model, base_body_name=base_body_name, robot_key=robot_key)

    def _spawn_debug_finish(out: tuple[float, float, float] | None, how: str) -> tuple[float, float, float] | None:
        _spawn_debug_emit_topdown(
            occ_map_dbg,
            clip_probe=clip_probe,
            occ_priority_xy=occ_xy,
            placed=out,
            how=how,
            initial_xy=initial_base_xy,
        )
        return out

    _pass_common = {
        "settle_kw": settle_kw,
        "zb_probe_bodies": zb_probe,
    }
    placed = _find_molmospaces_freejoint_xyz_pass(
        model,
        data,
        base_body_name=base_body_name,
        floor_effective=floor_effective,
        z_margins=z_margins,
        min_nonfloor_clearance=min_nonfloor_clearance,
        min_upward_clearance=float(_MIN_UPWARD_CEILING_CLEARANCE_M),
        max_geom_rbound=15.0,
        clip_erode_m=0.30,
        suppress_exterior_filter=False,
        xy_priority=occ_xy if occ_xy else None,
        **_pass_common,
    )
    if placed is not None:
        spawn_dbg(f"find: primary pass OK -> {placed}")
        return _spawn_debug_finish(placed, "primary")

    placed_relaxed = _find_molmospaces_freejoint_xyz_pass(
        model,
        data,
        base_body_name=base_body_name,
        floor_effective=floor_effective,
        z_margins=z_margins,
        min_nonfloor_clearance=min_nonfloor_clearance,
        min_upward_clearance=0.06,
        max_geom_rbound=45.0,
        clip_erode_m=0.12,
        suppress_exterior_filter=False,
        xy_priority=occ_xy if occ_xy else None,
        **_pass_common,
    )
    if placed_relaxed is not None:
        spawn_dbg(f"find: relaxed pass OK -> {placed_relaxed}")
        return _spawn_debug_finish(placed_relaxed, "relaxed")

    placed_ceiling_loose = _find_molmospaces_freejoint_xyz_pass(
        model,
        data,
        base_body_name=base_body_name,
        floor_effective=floor_effective,
        z_margins=z_margins,
        min_nonfloor_clearance=min_nonfloor_clearance,
        min_upward_clearance=0.035,
        max_geom_rbound=45.0,
        clip_erode_m=0.06,
        suppress_exterior_filter=False,
        xy_priority=occ_xy if occ_xy else None,
        **_pass_common,
    )
    if placed_ceiling_loose is not None:
        spawn_dbg(f"find: ceiling-loose pass OK -> {placed_ceiling_loose}")
        return _spawn_debug_finish(placed_ceiling_loose, "ceiling_loose")

    ray_exclude = int(base_bid) if base_bid >= 0 else -1
    placed_fb = _fallback_spawn_near_clip_center(
        model,
        data,
        base_body_name=base_body_name,
        floor_effective=floor_effective,
        robot_bodies=robot_bodies,
        ray_exclude=ray_exclude,
        xy_clip_raw=clip_probe,
        z_margins=z_margins,
        min_nonfloor_clearance=min_nonfloor_clearance,
        settle_kw=settle_kw,
        zb_probe_bodies=zb_probe,
    )
    if placed_fb is not None:
        spawn_dbg(f"find: clip-center fallback OK -> {placed_fb}")
        return _spawn_debug_finish(placed_fb, "clip_center_fallback")

    label = scene_label or "(unknown scene)"
    clip_s = "yes" if clip_probe is not None else "no"
    logger.warning(
        f"molmospaces_spawn.find_molmospaces_freejoint_xyz: primary, relaxed, ceiling-loose, and clip-center "
        f"fallback failed (scene={label!r} floor_geom_resolved={floor_effective!r} xy_clip_rect={clip_s})"
    )
    if restore_freejoint_base_from_model_qpos0(model, data, base_body_name=base_body_name):
        logger.info(
            "molmospaces_spawn.find_molmospaces_freejoint_xyz: all spawn passes failed; restored base "
            "free joint from model qpos0 (MJCF default) so the robot is not left at an internal "
            "search pose (e.g. hoisted z)."
        )
        spawn_dbg("find: failed_all_passes — restored base free joint from model qpos0 (MJCF default)")
    return _spawn_debug_finish(None, "failed_all_passes")
