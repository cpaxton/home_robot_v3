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
    _bodies_descending_from,
    _clamp_xy_into_rect,
    _coarse_grid_xy_in_clip,
    _erode_xy_rect,
    collision_scene_xy_clip_rect,
    effective_floor_geom_name,
    horizontal_spawn_rejects_exterior_tongue,
    iter_annulus_xy_candidates,
    robocasa_navigable_clearance_field,
    scene_collision_centroid_xy,
    upward_ray_hit_distance,
    walkable_floor_z_at_xy,
    worst_robot_nonfloor_contact_dist,
)
from emet.simulation.spawn_planar import planar_spawn_footprint_xy_margin_m
from emet.simulation.spawn_settle import (
    _molmospaces_z_settle_options,
    _post_settle_pose_acceptable,
    restore_freejoint_base_from_model_qpos0,
    settle_free_base_z_to_floor,
    write_freejoint_base_xyzw,
)


def _robocasa_autoplace_enabled(robocasa_autoplace_env: str | None = None) -> bool:
    raw = (
        robocasa_autoplace_env
        if robocasa_autoplace_env is not None
        else os.environ.get("EMET_ROBOSUITE_AUTOPLACE", "1")
    )
    v = str(raw).strip().lower()
    return v not in ("0", "false", "no", "off")


def want_robocasa_planar_autoplace(
    *,
    environment: dict[str, Any] | None,
    robot_spec: RobotSpec,
    robocasa_autoplace_env: str | None = None,
) -> bool:
    """Whether to run collision-free **planar** base placement for Robocasa kitchens.

    ``EMET_ROBOSUITE_AUTOPLACE`` (default ``1``): ``0``/``false``/``no``/``off`` disables.
    Requires ``environment.kind == \"robocasa\"`` and ``robot_spec.planar_base_joint_names`` (length 3).
    """
    if environment is None or environment.get("kind") != "robocasa":
        return False
    pbn = getattr(robot_spec, "planar_base_joint_names", None)
    if not pbn or len(pbn) != 3:
        return False
    return _robocasa_autoplace_enabled(robocasa_autoplace_env)


def want_robocasa_freejoint_autoplace(
    *,
    environment: dict[str, Any] | None,
    robot_spec: RobotSpec,
    robocasa_autoplace_env: str | None = None,
) -> bool:
    """Whether to run collision-free **freejoint** base placement for Robocasa kitchens.

    Used for robots such as Stretch, Galaxea R1 / RB-Y1 (``freejoint`` base, no planar slide joints).
    Planar robots (e.g. innate_mars) use :func:`want_robocasa_planar_autoplace`.
    """
    if environment is None or environment.get("kind") != "robocasa":
        return False
    pbn = getattr(robot_spec, "planar_base_joint_names", None)
    if pbn and len(pbn) == 3:
        return False
    if not _robocasa_autoplace_enabled(robocasa_autoplace_env):
        return False
    return bool(getattr(robot_spec, "base_link_name", None))


def _quat_wxyz_from_planar_yaw(yaw: float) -> tuple[float, float, float, float]:
    wt = float(yaw)
    return (float(math.cos(wt * 0.5)), 0.0, 0.0, float(math.sin(wt * 0.5)))


def _robocasa_backward_bias_xy(hx: float, hy: float, yaw: float) -> list[tuple[float, float]]:
    """Priority XY samples along −forward from a Robosuite spawn hint (counters / islands)."""
    fx = float(math.cos(yaw))
    fy = float(math.sin(yaw))
    return [(hx - d * fx, hy - d * fy) for d in (0.15, 0.25, 0.35, 0.45, 0.55)]


def _try_robocasa_freejoint_at_xy_yaw(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    floor_effective: str,
    robot_bodies: set[int],
    ray_exclude: int,
    x: float,
    y: float,
    yaw: float,
    z_margins: tuple[float, ...],
    min_nonfloor_clearance: float,
    min_upward_clearance: float,
    xy_clip_scene: tuple[float, float, float, float] | None,
    clip_edge_pad_m: float,
    clip_guard_body_names: tuple[str, ...],
    clip_guard_pad_m: float,
    settle_kw: dict[str, float] | None = None,
    zb_probe_bodies: set[int] | None = None,
) -> tuple[float, float, float] | None:
    """Single Robocasa freejoint (x, y, yaw): floor probe, contact ladder, clip inset."""
    quat = _quat_wxyz_from_planar_yaw(yaw)
    _settle_kw = dict(settle_kw or {})
    z_air = max(8.0, 3.0 * float(model.stat.extent))
    if not write_freejoint_base_xyzw(
        model, data, base_body_name=base_body_name, x=float(x), y=float(y), z=z_air, quat_wxyz=quat
    ):
        return None
    mujoco.mj_forward(model, data)
    z_floor = walkable_floor_z_at_xy(model, data, x, y, floor_geom_name=floor_effective, exclude_body_id=ray_exclude)
    if z_floor is None:
        return None
    z_probe = float(z_floor) + 0.08
    up_dist = upward_ray_hit_distance(model, data, x, y, z_probe, exclude_body_id=ray_exclude)
    if min_upward_clearance >= 0.0 and up_dist is not None and up_dist < min_upward_clearance:
        return None
    if horizontal_spawn_rejects_exterior_tongue(model, data, x, y, z_probe, exclude_body_id=ray_exclude):
        return None
    for zm in z_margins:
        z = z_floor + float(zm)
        if not write_freejoint_base_xyzw(model, data, base_body_name=base_body_name, x=x, y=y, z=z, quat_wxyz=quat):
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
            if xy_clip_scene is not None:
                base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
                if base_bid >= 0:
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
            return (float(x), float(y), z_settled)
    return None


def find_robocasa_freejoint_xyz(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    robot_spec: RobotSpec,
    floor_geom_name: str = "floor",
    scene_label: str | None = None,
    merged_mjcf_path: str | None = None,
    environment: dict[str, Any] | None = None,
    spawn_hint_xyt: np.ndarray | None = None,
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
    robocasa_first_clearance_m: float | None = None,
) -> tuple[float, float, float] | None:
    """Collision-free world (x, y, z) for a Robocasa kitchen freejoint base (Stretch, Galaxea R1, …).

    Tries ``spawn_hint_xyt`` (Robosuite ``init_robot_base_pos``) first, then backward-biased and
    annulus/grid candidates inside the eroded walkable clip. Yaw follows the hint when present,
    with π/4 fallbacks. Open ceilings are allowed (Robocasa profile). Returns ``None`` on failure
    and restores ``model.qpos0`` on the base free joint.
    """
    floor_effective = effective_floor_geom_name(model, floor_geom_name)
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    robot_bodies: set[int] = set()
    if base_bid >= 0:
        robot_bodies = _bodies_descending_from(model, base_bid)
    ray_exclude = int(base_bid) if base_bid >= 0 else -1

    margin = planar_spawn_footprint_xy_margin_m(robot_spec)
    clip_pad = robot_spec.planar_spawn_clip_edge_pad_m
    if clip_pad is None:
        clip_pad = float(0.22 + 0.5 * float(robot_spec.planar_spawn_xy_extra_margin_m))
    guard_names = robot_spec.planar_spawn_clip_guard_body_names
    if not guard_names and robot_spec.planar_spawn_clip_guard_body_name:
        guard_names = (robot_spec.planar_spawn_clip_guard_body_name,)
    guard_pad = float(robot_spec.planar_spawn_clip_guard_pad_m)

    if robocasa_first_clearance_m is None:
        robocasa_first_clearance_m = robot_spec.planar_spawn_robocasa_first_clearance_m

    # When a Robocasa first-clearance is configured, do not return the first near-hint
    # hit: prefer the open-floor pose with larger contact clearance (tie-break: nearer hint).
    prefer_open_floor = robocasa_first_clearance_m is not None

    hint_xy: tuple[float, float] | None = None
    hint_yaw: float | None = None
    if spawn_hint_xyt is not None:
        h = np.asarray(spawn_hint_xyt, dtype=np.float64).reshape(-1)
        if h.size >= 2:
            hint_xy = (float(h[0]), float(h[1]))
            if h.size >= 3:
                hint_yaw = float(h[2])
    if hint_xy is None and isinstance(environment, dict):
        raw_hint = environment.get("spawn_hint_xyt")
        if raw_hint is not None:
            h = np.asarray(raw_hint, dtype=np.float64).reshape(-1)
            if h.size >= 2:
                hint_xy = (float(h[0]), float(h[1]))
                if h.size >= 3:
                    hint_yaw = float(h[2])
    if hint_yaw is None and base_bid >= 0:
        rot = data.body(base_body_name).xmat.reshape(3, 3)
        hint_yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))

    xy_clip_scene = collision_scene_xy_clip_rect(
        model,
        data,
        robot_bodies=robot_bodies,
        floor_geom_name=floor_effective,
        margin=0.55,
        max_geom_rbound=15.0,
        suppress_exterior_filter=False,
    )
    inset = float(margin) + 0.10
    xy_clip = _erode_xy_rect(xy_clip_scene, inset) if xy_clip_scene is not None else None
    if xy_clip is None:
        xy_clip = xy_clip_scene

    if hint_xy is not None:
        ox, oy = hint_xy
    elif xy_clip is not None:
        ox = 0.5 * float(xy_clip[0] + xy_clip[1])
        oy = 0.5 * float(xy_clip[2] + xy_clip[3])
    else:
        centroid = scene_collision_centroid_xy(
            model,
            data,
            robot_bodies=robot_bodies,
            floor_geom_name=floor_effective,
            max_geom_rbound=15.0,
            suppress_exterior_filter=False,
        )
        if centroid is not None:
            ox, oy = float(centroid[0]), float(centroid[1])
        else:
            ox, oy = 0.0, 0.0
    if xy_clip is not None:
        ox, oy = _clamp_xy_into_rect(ox, oy, xy_clip)

    r_annulus_max = 3.5
    if xy_clip is not None:
        x0c, x1c, y0c, y1c = xy_clip
        half_diag = 0.5 * math.hypot(float(x1c - x0c), float(y1c - y0c))
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
        for px, py in _coarse_grid_xy_in_clip(xy_clip, step=0.48, max_points=520):
            key = (round(px, 2), round(py, 2))
            if key in seen:
                continue
            seen.add(key)
            base_candidates.append((float(px), float(py)))

    priority_xy: list[tuple[float, float]] = []
    if hint_xy is not None:
        priority_xy.append(hint_xy)
        if hint_yaw is not None:
            priority_xy.extend(_robocasa_backward_bias_xy(hint_xy[0], hint_xy[1], hint_yaw))
    # Dedupe against the broad base set *without* pre-seeding from it, so base candidates can
    # be appended to priority_xy for the Robocasa open-floor search (needed to move the spawn
    # off a counter-adjacent hint that has no A*-navigable path out).
    seen_xy = {(round(a, 2), round(b, 2)) for a, b in priority_xy}
    for px, py in base_candidates:
        k = (round(px, 2), round(py, 2))
        if k in seen_xy:
            continue
        seen_xy.add(k)
        priority_xy.append((float(px), float(py)))
    # Robocasa open-floor profile: also allow the broad annulus/grid candidate set so the
    # navigability gate can move the spawn away from a counter-adjacent hint. Otherwise the
    # hint corridor (hint + backward bias) can trap the robot against a wall or counter with
    # no A*-navigable path out.
    if prefer_open_floor:
        candidates = priority_xy  # hint corridor first, then broad annulus/grid candidates
    elif hint_xy is not None:
        candidates = priority_xy
        hx, hy = hint_xy
        candidates.sort(key=lambda p: (p[0] - hx) ** 2 + (p[1] - hy) ** 2)
    else:
        candidates = base_candidates

    raw_xy_cap = os.environ.get("EMET_PLANAR_SPAWN_MAX_XY", "").strip()
    max_xy_candidates = int(raw_xy_cap) if raw_xy_cap.isdigit() else (1200 if prefer_open_floor else 400)
    spawn_dbg(
        f"robocasa_freejoint_find: n_base_candidates={len(base_candidates)} "
        f"n_priority={len(priority_xy)} r_annulus_max={r_annulus_max:.2f} "
        f"xy_clip={'yes' if xy_clip is not None else 'no'}"
    )
    if max_xy_candidates > 0 and len(candidates) > max_xy_candidates:
        candidates = candidates[:max_xy_candidates]

    yaws = [float(k * math.pi / 4.0) for k in range(8)]
    if hint_yaw is not None:
        hyaw = float(hint_yaw)
        yaws = [hyaw] + [y for y in yaws if abs(float(np.arctan2(np.sin(y - hyaw), np.cos(y - hyaw)))) > 0.02]

    clearance_passes: tuple[tuple[float, str], ...] = (
        (0.045, "clear045"),
        (0.028, "clear028"),
        (0.014, "clear014"),
        (0.005, "clear005"),
        (-5e-5, "clear_num"),
    )
    if robocasa_first_clearance_m is not None:
        fc = float(robocasa_first_clearance_m)
        clearance_passes = (
            (fc, "clear_robo_strict"),
            (max(0.028, 0.55 * fc), "clear_robo_mid"),
            (max(0.014, 0.32 * fc), "clear_robo_lo"),
            (0.005, "clear005"),
            (-5e-5, "clear_num"),
        )

    settle_kw, zb_probe = _molmospaces_z_settle_options(
        model, base_body_name=base_body_name, robot_key=str(robot_spec.name)
    )
    min_upward = -1.0

    # True A*-style navigability gate. Contact distance alone saturates at 1.0 (no contact) so
    # the old "open floor" score happily chose a pose 0.5 m from a room wall, where the planner's
    # padded clearance field never reaches min_clearance and every explore plan is rejected.
    # Build this *before* the candidate loop (candidate breadth depends on prefer_open_floor).
    nav_field: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float]] | None = None
    nav_area_map: np.ndarray | None = None
    if prefer_open_floor and xy_clip_scene is not None:
        nav_field = robocasa_navigable_clearance_field(
            model,
            data,
            floor_geom_name=floor_effective,
            xy_rect=xy_clip_scene,
            min_clearance_m=float(robot_spec.footprint.width) * 0.5 + 0.05,
            pad_m=0.20,
            grid_resolution_m=0.10,
        )
        if nav_field is not None:
            nav, _clr, labels, _rect = nav_field
            import scipy.ndimage as _ndi

            _nlab = int(labels.max())
            nav_area_map = _ndi.sum(np.ones_like(nav), labels, index=range(1, _nlab + 1))
            spawn_dbg(
                f"robocasa_freejoint_find: navigable regions={_nlab} "
                f"cells={int(nav.sum())} area_m2={float(nav.sum()) * 0.01:.1f}"
            )

    def _nav_lookup(x: float, y: float) -> tuple[float, int]:
        """Return ``(navigable-region area m2, 1 if navigable)`` for world ``(x, y)``."""
        if nav_field is None or nav_area_map is None:
            return 0.0, 1
        nav, _clr, labels, rect = nav_field
        # Rect convention is ``(xmin, xmax, ymin, ymax)`` (see collision_scene_xy_clip_rect).
        x0n, _x1n, y0n, _y1n = rect
        res = 0.10
        i = int(round((float(x) - x0n) / res))
        j = int(round((float(y) - y0n) / res))
        nx_, ny_ = nav.shape
        if not (0 <= i < nx_ and 0 <= j < ny_):
            return 0.0, 0
        if not bool(nav[i, j]):
            return 0.0, 0
        label = int(labels[i, j])
        if label <= 0 or label - 1 >= len(nav_area_map):
            return 0.0, 0
        return float(nav_area_map[label - 1]), 1

    spawn_dbg(
        f"robocasa_freejoint_find: scene={scene_label!r} n_xy={len(candidates)} "
        f"margin_m={margin:.3f} hint={'yes' if hint_xy else 'no'} "
        f"prefer_open_floor={prefer_open_floor} first_cands={[(round(a, 2), round(b, 2)) for a, b in candidates[:6]]}"
    )
    # Require the winning pose to sit in a navigable region of at least this area so OVMM
    # explore actually has room to move (not a 1-cell pocket next to a wall).
    min_nav_area_m2 = float(os.environ.get("EMET_ROBOCASA_SPAWN_MIN_NAV_AREA_M2", "1.5"))
    if prefer_open_floor and nav_field is not None and nav_area_map is not None:
        prefiltered = [
            p
            for p in candidates
            if _nav_lookup(float(p[0]), float(p[1]))[1] == 1
            and _nav_lookup(float(p[0]), float(p[1]))[0] >= min_nav_area_m2
        ]
        spawn_dbg(f"robocasa_freejoint_find: prefilter navigable candidates {len(prefiltered)}/{len(candidates)}")
        if prefiltered:
            # Prefer open floor: larger navigable region, then closer to the walkable-clip
            # centroid (away from counters / walls), then nearer the hint.
            cx = 0.5 * (float(xy_clip[0]) + float(xy_clip[1])) if xy_clip is not None else 0.0
            cy = 0.5 * (float(xy_clip[2]) + float(xy_clip[3])) if xy_clip is not None else 0.0
            hx = float(hint_xy[0]) if hint_xy is not None else 0.0
            hy = float(hint_xy[1]) if hint_xy is not None else 0.0

            def _open_floor_key(p: tuple[float, float]) -> tuple[float, float, float]:
                na = _nav_lookup(float(p[0]), float(p[1]))[0]
                dc = math.hypot(float(p[0]) - cx, float(p[1]) - cy)
                dh = math.hypot(float(p[0]) - hx, float(p[1]) - hy)
                return (-na, dc, dh)

            candidates = sorted(prefiltered, key=_open_floor_key)
        else:
            # No candidate is navigable on the full grid; fall back to the hint corridor so we
            # still return *something* (the sim will spawn contact-free even if explore is tight).
            candidates = [p for p in candidates if abs(float(p[0]) - (hint_xy[0] if hint_xy else 0.0)) < 1.0]
            spawn_dbg(f"robocasa_freejoint_find: no navigable candidates; hint-corridor fallback n={len(candidates)}")

    for min_clear, tag in clearance_passes:
        # With the navigable prefilter, all candidates already satisfy the open-floor gate, so
        # return the first pose that also passes the contact-clearance placement (no full scan).
        if prefer_open_floor and nav_field is not None and nav_area_map is not None:
            for x, y in candidates:
                for yaw in yaws:
                    placed = _try_robocasa_freejoint_at_xy_yaw(
                        model,
                        data,
                        base_body_name=base_body_name,
                        floor_effective=floor_effective,
                        robot_bodies=robot_bodies,
                        ray_exclude=ray_exclude,
                        x=float(x),
                        y=float(y),
                        yaw=float(yaw),
                        z_margins=z_margins,
                        min_nonfloor_clearance=float(min_clear),
                        min_upward_clearance=min_upward,
                        xy_clip_scene=xy_clip_scene,
                        clip_edge_pad_m=float(clip_pad),
                        clip_guard_body_names=tuple(str(n) for n in guard_names),
                        clip_guard_pad_m=guard_pad,
                        settle_kw=settle_kw,
                        zb_probe_bodies=zb_probe,
                    )
                    if placed is not None:
                        nav_area = _nav_lookup(float(x), float(y))[0]
                        spawn_dbg(
                            f"robocasa_freejoint_find: OK pass={tag!r} xy=({x:.3f},{y:.3f}) "
                            f"yaw={yaw:.3f} nav_area={nav_area:.2f}m2"
                        )
                        return placed
            continue
        best: tuple[tuple[float, float], tuple[float, float, float], float, float, float] | None = None
        for x, y in candidates:
            for yaw in yaws:
                placed = _try_robocasa_freejoint_at_xy_yaw(
                    model,
                    data,
                    base_body_name=base_body_name,
                    floor_effective=floor_effective,
                    robot_bodies=robot_bodies,
                    ray_exclude=ray_exclude,
                    x=float(x),
                    y=float(y),
                    yaw=float(yaw),
                    z_margins=z_margins,
                    min_nonfloor_clearance=float(min_clear),
                    min_upward_clearance=min_upward,
                    xy_clip_scene=xy_clip_scene,
                    clip_edge_pad_m=float(clip_pad),
                    clip_guard_body_names=tuple(str(n) for n in guard_names),
                    clip_guard_pad_m=guard_pad,
                    settle_kw=settle_kw,
                    zb_probe_bodies=zb_probe,
                )
                if placed is None:
                    continue
                if not prefer_open_floor:
                    spawn_dbg(f"robocasa_freejoint_find: OK pass={tag!r} xy=({x:.3f},{y:.3f}) yaw={yaw:.3f}")
                    return placed
                worst = float(
                    worst_robot_nonfloor_contact_dist(
                        model,
                        data,
                        base_body_name=base_body_name,
                        floor_geom_name=floor_effective,
                    )
                )
                nav_area, nav_ok = _nav_lookup(float(x), float(y))
                if nav_ok != 1 or nav_area < min_nav_area_m2:
                    spawn_dbg(
                        f"robocasa_freejoint_find: skip pass={tag!r} xy=({x:.3f},{y:.3f}) "
                        f"nav_area={nav_area:.2f}m2 (need >= {min_nav_area_m2:.1f})"
                    )
                    continue
                if hint_xy is not None:
                    d_hint = float(math.hypot(float(x) - hint_xy[0], float(y) - hint_xy[1]))
                else:
                    d_hint = 0.0
                # Prefer genuinely open floor: navigable region size first, then contact
                # clearance, then closer to the walkable-clip centroid (hint is a soft tie-break).
                if xy_clip is not None:
                    cx = 0.5 * (float(xy_clip[0]) + float(xy_clip[1]))
                    cy = 0.5 * (float(xy_clip[2]) + float(xy_clip[3]))
                    d_cent = float(math.hypot(float(x) - cx, float(y) - cy))
                else:
                    d_cent = 0.0
                score = (nav_area, worst, -d_cent, -d_hint)
                if best is None or score > best[0]:
                    best = (score, placed, float(x), float(y), float(yaw))
        if best is not None:
            _score, placed_best, bx, by, byaw = best
            # Re-apply winner so ``data`` / freejoint match the returned pose.
            placed_final = _try_robocasa_freejoint_at_xy_yaw(
                model,
                data,
                base_body_name=base_body_name,
                floor_effective=floor_effective,
                robot_bodies=robot_bodies,
                ray_exclude=ray_exclude,
                x=bx,
                y=by,
                yaw=byaw,
                z_margins=z_margins,
                min_nonfloor_clearance=float(min_clear),
                min_upward_clearance=min_upward,
                xy_clip_scene=xy_clip_scene,
                clip_edge_pad_m=float(clip_pad),
                clip_guard_body_names=tuple(str(n) for n in guard_names),
                clip_guard_pad_m=guard_pad,
                settle_kw=settle_kw,
                zb_probe_bodies=zb_probe,
            )
            out = placed_final if placed_final is not None else placed_best
            spawn_dbg(
                f"robocasa_freejoint_find: OK pass={tag!r} xy=({bx:.3f},{by:.3f}) yaw={byaw:.3f} "
                f"nav_area={_score[0]:.2f}m2 worst={_score[1]:.4f} d_cent={-_score[2]:.3f}"
            )
            return out

    spawn_dbg(f"robocasa_freejoint_find: failed scene={scene_label!r}")
    if restore_freejoint_base_from_model_qpos0(model, data, base_body_name=base_body_name):
        spawn_dbg("robocasa_freejoint_find: restored base from model qpos0")
    return None
