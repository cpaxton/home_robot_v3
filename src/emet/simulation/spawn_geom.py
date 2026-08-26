# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from typing import Any

import mujoco
import numpy as np

import emet.utils.logger as log

logger = log.Logger(__name__)

# Reject XY where an upward ray finds nothing (open void) or only a glancing hit very close above
# the probe (likely under a thin shelf / numerical noise).
_MIN_UPWARD_CEILING_CLEARANCE_M = 0.15

# Horizontal cardinal rays: orthographic occupancy can mark infinite exterior floor as “free”.
# We skip XY that look like an **exterior tongue**: many open cardinals, or two opens with no
# distant wall in any remaining direction (tight cage + voids on multiple sides).
_HORIZ_EXTERIOR_MAX_FINITE_M = 0.75

# Default caps (override with env): occupancy free-cell priority count, coarse XY grid in clip.
_DEFAULT_MOLMOSPACES_OCC_PRIORITY_MAX = 900
_DEFAULT_MOLMOSPACES_GRID_IN_CLIP_MAX = 420
_DEFAULT_MOLMOSPACES_GRID_STEP_M = 0.46


def _static_collision_occupancy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    floor_geom_name: str,
    grid_resolution_m: float,
    xy_rect: tuple[float, float, float, float],
    obstacle_band: tuple[float, float] = (0.2, 1.5),
) -> np.ndarray | None:
    """Top-down occupancy of scene collision geoms in the walkable band.

    Marks cells covered by any collision geom (``contype``/``conaffinity`` non-zero) whose body is
    **not** the robot and whose surface lies in the *obstacle_band* (0.2–1.5 m). Geom footprint is
    approximated by its world position + bounding radius, so tall walls/counters register without
    ray-casting pitfalls (vertical rays hit ceiling-level shells first). Mirrors what the head
    camera voxel map builds so spawn validation matches planner clearance.
    """
    floor_eff = effective_floor_geom_name(model, floor_geom_name)
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_eff)
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    robot_bodies: set[int] = set()
    if base_bid >= 0:
        for b in range(model.nbody):
            x = int(b)
            while x > 0:
                if x == base_bid:
                    robot_bodies.add(int(b))
                    break
                x = int(model.body_parentid[x])
    res = float(grid_resolution_m)
    x0, x1, y0, y1 = xy_rect
    nx = max(1, int(round((x1 - x0) / res)))
    ny = max(1, int(round((y1 - y0) / res)))
    occ = np.zeros((nx, ny), dtype=bool)
    band_lo = float(obstacle_band[0])
    # Geoms whose entire surface sits above the mobile base's navigable height do not block
    # navigation (Robocasa room boundary shells live at z≈1.5). Only count geoms that reach
    # down into the band.
    nav_ceiling_m = float(os.environ.get("EMET_ROBOCASA_NAV_CEILING_M", "1.2"))
    for g in range(model.ngeom):
        if g == floor_gid:
            continue
        if int(model.geom_bodyid[g]) in robot_bodies:
            continue
        if int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0:
            continue
        if int(model.geom_type[g]) == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        p = data.geom_xpos[g]
        sz = model.geom_size[g]
        # Lowest / highest point of the geom surface (approx. from size + small mesh margin).
        z_lo = float(p[2]) - (float(sz[2]) if sz.size >= 3 else 0.0) - 0.05
        z_hi = float(p[2]) + (float(sz[2]) if sz.size >= 3 else 0.0) + 0.05
        if z_lo > nav_ceiling_m:
            # Entire geom hangs above navigation height (room shells / upper cabinets).
            continue
        if z_hi < band_lo:
            # Entire geom sits below the obstacle band (floor slab / backing).
            continue
        rb = float(model.geom_rbound[g])
        if rb <= 0.0:
            continue
        # Use the horizontal extent (half-diagonal) rather than the 3D rbound so tall thin
        # walls do not mark cells many meters away.
        rad = float(np.hypot(sz[0], sz[1])) if sz.size >= 2 else rb
        rad = min(max(rad, 0.05), 1.0)
        i0 = max(0, int(np.floor((float(p[0]) - rad - x0) / res)))
        i1 = min(nx - 1, int(np.ceil((float(p[0]) + rad - x0) / res)))
        j0 = max(0, int(np.floor((float(p[1]) - rad - y0) / res)))
        j1 = min(ny - 1, int(np.ceil((float(p[1]) + rad - y0) / res)))
        if i1 >= i0 and j1 >= j0:
            occ[i0 : i1 + 1, j0 : j1 + 1] = True
    return occ


def robocasa_navigable_clearance_field(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    floor_geom_name: str,
    xy_rect: tuple[float, float, float, float],
    min_clearance_m: float = 0.22,
    pad_m: float = 0.20,
    grid_resolution_m: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float]] | None:
    """Return ``(navigable, clearance_m, labels, rect)`` for static scene geometry.

    Occupancy is sampled at *grid_resolution_m*, obstacles are padded by *pad_m* (matching the
    planner's ``pad_obstacles``), then an EDT clearance field is computed. A cell is *navigable*
    when free and ``clearance >= min_clearance_m`` — the same gate the A* planner applies to
    voxel map obstacles. ``labels`` gives connected navigable regions for area checks.
    """
    occ = _static_collision_occupancy(
        model,
        data,
        floor_geom_name=floor_geom_name,
        grid_resolution_m=grid_resolution_m,
        xy_rect=xy_rect,
    )
    if occ is None:
        return None
    try:
        from scipy import ndimage as _ndi
        from skimage.morphology import disk as _disk

        pad_cells = int(round(float(pad_m) / float(grid_resolution_m)))
        occ_p = _ndi.binary_dilation(occ, structure=_disk(max(1, pad_cells)))
        clr = _ndi.distance_transform_edt(~occ_p) * float(grid_resolution_m)
        clr[occ_p] = 0.0
        labels, _nlab = _ndi.label((~occ_p) & (clr >= float(min_clearance_m)))
    except Exception as exc:  # pragma: no cover - optional scipy/skimage path
        logger.warning(f"robocasa navigable clearance skipped ({exc!r}).")
        return None
    nav = (~occ_p) & (clr >= float(min_clearance_m))
    return nav, clr, labels, (float(xy_rect[0]), float(xy_rect[1]), float(xy_rect[2]), float(xy_rect[3]))


def resolve_floor_geom_name(model: mujoco.MjModel) -> str | None:
    """Best-effort walkable floor plane for iTHOR / Molmo / generic MJCF (not always named ``floor``)."""
    for name in ("floor", "Floor", "ground", "Ground"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0:
            return name
    for g in range(model.ngeom):
        if int(model.geom_type[g]) != mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if not (int(model.geom_contype[g]) or int(model.geom_conaffinity[g])):
            continue
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if nm:
            return str(nm)
    # Robocasa kitchens often use a large **visual-only** floor mesh/plane (e.g. ``floor_room_g0_vis``);
    # skipping these made ``walkable_floor_z_at_xy`` bail out and planar spawn never find a pose.
    for g in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if not nm:
            continue
        low = nm.lower()
        if "floor_room" in low or low.startswith("floor") and low.endswith("_vis"):
            return str(nm)
    return None


def effective_floor_geom_name(model: mujoco.MjModel, floor_geom_name: str = "floor") -> str:
    """Return *floor_geom_name* if present, else :func:`resolve_floor_geom_name` or the hint string."""
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_geom_name) >= 0:
        return floor_geom_name
    resolved = resolve_floor_geom_name(model)
    return resolved if resolved is not None else floor_geom_name


def _bodies_descending_from(model: mujoco.MjModel, root_body_id: int) -> set[int]:
    """All body ids whose kinematic chain includes *root_body_id* (including the root)."""
    out: set[int] = set()
    for b in range(model.nbody):
        x = b
        guard = 0
        while x >= 0 and guard < model.nbody + 2:
            guard += 1
            if x == root_body_id:
                out.add(b)
                break
            x = int(model.body_parentid[x])
    return out


def _geom_body_is_robot(model: mujoco.MjModel, geom_id: int, robot_bodies: set[int]) -> bool:
    return int(model.geom_bodyid[geom_id]) in robot_bodies


_EXTERIOR_GEOM_NAME_SUBSTR = (
    "porch",
    "deck",
    "patio",
    "exterior",
    "outdoor",
    "fence",
    "fencing",
    "bannister",
    "railing",
)


def _geom_exterior_heuristic(model: mujoco.MjModel, geom_id: int) -> bool:
    """True if geom / body name suggests outdoor porch / deck (best-effort for iTHOR-style MJCF)."""
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if name:
        low = name.lower()
        if any(s in low for s in _EXTERIOR_GEOM_NAME_SUBSTR):
            return True
    bid = int(model.geom_bodyid[geom_id])
    bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    if bname:
        low = bname.lower()
        if any(s in low for s in _EXTERIOR_GEOM_NAME_SUBSTR):
            return True
    return False


def collision_scene_xy_clip_rect(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    robot_bodies: set[int],
    floor_geom_name: str = "floor",
    margin: float = 0.42,
    max_geom_rbound: float = 15.0,
    suppress_exterior_filter: bool = False,
) -> tuple[float, float, float, float] | None:
    """Rough axis-aligned (x,y) range of scene collision content (excludes robot and infinite floor).

    Uses each collision geom's world position and ``geom_rbound`` so we do not sample spawn XY
    on the ``floor`` plane beyond walls/room content (which would look like a black void).

    Geoms with a very large ``geom_rbound`` (typical merged room / house meshes) **cap** their
    contribution at *max_geom_rbound* instead of being skipped entirely. Skipping them used to
    yield an empty clip, then XY search fell back to world ``(0,0)`` with no clip — spawns on open
    floor **outside** offset house footprints.
    """
    floor_resolved = effective_floor_geom_name(model, floor_geom_name)
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_resolved)
    xmin: float = 1e30
    xmax: float = -1e30
    ymin: float = 1e30
    ymax: float = -1e30
    any_hit = False
    for g in range(model.ngeom):
        if floor_gid >= 0 and g == floor_gid:
            continue
        if int(model.geom_type[g]) == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if not (int(model.geom_contype[g]) or int(model.geom_conaffinity[g])):
            continue
        if _geom_body_is_robot(model, g, robot_bodies):
            continue
        if not suppress_exterior_filter and _geom_exterior_heuristic(model, g):
            continue
        rb = float(model.geom_rbound[g])
        if rb <= 0.0:
            continue
        rb = min(rb, float(max_geom_rbound))
        p = data.geom_xpos[g]
        xmin = min(xmin, float(p[0]) - rb)
        xmax = max(xmax, float(p[0]) + rb)
        ymin = min(ymin, float(p[1]) - rb)
        ymax = max(ymax, float(p[1]) + rb)
        any_hit = True
    if not any_hit or xmax - xmin < 1.2 or ymax - ymin < 1.2:
        return None
    return (
        xmin + float(margin),
        xmax - float(margin),
        ymin + float(margin),
        ymax - float(margin),
    )


def _erode_xy_rect(rect: tuple[float, float, float, float], inset: float) -> tuple[float, float, float, float] | None:
    """Shrink a clip rectangle symmetrically; return ``None`` if it would collapse."""
    x0, x1, y0, y1 = rect
    if x1 - x0 < 2.0 * inset + 0.9 or y1 - y0 < 2.0 * inset + 0.9:
        return None
    return (x0 + inset, x1 - inset, y0 + inset, y1 - inset)


def _xy_rect_area_m2(rect: tuple[float, float, float, float] | None) -> float | None:
    if rect is None:
        return None
    x0, x1, y0, y1 = rect
    w = float(x1) - float(x0)
    h = float(y1) - float(y0)
    if w <= 0.0 or h <= 0.0:
        return 0.0
    return w * h


def compute_spawn_walkable_map_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    floor_geom_name: str = "floor",
    grid_resolution_m: float = 0.05,
    footprint_xy_margin_m: float = 0.35,
    spawn_profile: str = "robocasa",
) -> dict[str, Any]:
    """Walkable floor map from spawner collision clip (same geometry as Robocasa autoplace).

    Samples a top-down grid inside the eroded scene clip and counts cells where
    :func:`walkable_floor_z_at_xy` resolves a floor height (matches spawner occupancy logic).
    """
    floor_effective = effective_floor_geom_name(model, floor_geom_name)
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    robot_bodies: set[int] = set()
    if base_bid >= 0:
        robot_bodies = _bodies_descending_from(model, base_bid)
    ray_excl = int(base_bid) if base_bid >= 0 else -1

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

    grid_res = float(grid_resolution_m)
    walkable_cells = 0
    if xy_clip is not None:
        x0, x1, y0, y1 = xy_clip
        xs = np.arange(float(x0), float(x1) + grid_res * 0.5, grid_res)
        ys = np.arange(float(y0), float(y1) + grid_res * 0.5, grid_res)
        for x in xs:
            for y in ys:
                zf = walkable_floor_z_at_xy(
                    model,
                    data,
                    float(x),
                    float(y),
                    floor_geom_name=floor_effective,
                    exclude_body_id=ray_excl,
                )
                if zf is not None:
                    walkable_cells += 1

    cell_area = grid_res * grid_res
    scene_area = _xy_rect_area_m2(xy_clip_scene)
    scene_cells = int(round(scene_area / cell_area)) if scene_area is not None else 0
    return {
        "grid_resolution_m": grid_res,
        "clip_scene_xy": list(xy_clip_scene) if xy_clip_scene is not None else None,
        "clip_eroded_xy": list(xy_clip) if xy_clip is not None else None,
        "clip_scene_area_m2": scene_area,
        "clip_eroded_area_m2": _xy_rect_area_m2(xy_clip),
        "spawn_walkable_cell_count": int(walkable_cells),
        "spawn_walkable_area_m2": float(walkable_cells * cell_area),
        "scene_walkable_cell_count": scene_cells,
        "scene_walkable_area_m2": float(scene_cells * cell_area) if scene_cells else scene_area,
        "spawn_profile": spawn_profile,
    }


def _clamp_xy_into_rect(x: float, y: float, rect: tuple[float, float, float, float]) -> tuple[float, float]:
    """Clamp *x*, *y* to the closed axis-aligned rectangle (xmin, xmax, ymin, ymax)."""
    x0, x1, y0, y1 = rect
    return (min(max(float(x), x0), x1), min(max(float(y), y0), y1))


def _max_xy_distance_to_rect_corners(x: float, y: float, rect: tuple[float, float, float, float]) -> float:
    """Max Euclidean distance from *(x,y)* to one of the four corners of *rect*."""
    x0, x1, y0, y1 = rect
    corners = ((x0, y0), (x1, y0), (x0, y1), (x1, y1))
    return max(math.hypot(float(x) - cx, float(y) - cy) for cx, cy in corners)


def scene_collision_centroid_xy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    robot_bodies: set[int],
    floor_geom_name: str = "floor",
    max_geom_rbound: float = 15.0,
    suppress_exterior_filter: bool = False,
) -> tuple[float, float] | None:
    """Mean (x, y) of scene collision geoms (same filters as :func:`collision_scene_xy_clip_rect`).

    Like the clip rect, each geom's influence on the mean is capped at *max_geom_rbound* so merged
    mega-meshes still contribute a centroid near the house rather than being dropped.
    """
    floor_resolved = effective_floor_geom_name(model, floor_geom_name)
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_resolved)
    sx = 0.0
    sy = 0.0
    n = 0
    for g in range(model.ngeom):
        if floor_gid >= 0 and g == floor_gid:
            continue
        if int(model.geom_type[g]) == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if not (int(model.geom_contype[g]) or int(model.geom_conaffinity[g])):
            continue
        if _geom_body_is_robot(model, g, robot_bodies):
            continue
        if not suppress_exterior_filter and _geom_exterior_heuristic(model, g):
            continue
        rb = float(model.geom_rbound[g])
        if rb <= 0.0:
            continue
        rb = min(rb, float(max_geom_rbound))
        p = data.geom_xpos[g]
        sx += float(p[0])
        sy += float(p[1])
        n += 1
    if n == 0:
        return None
    return (sx / n, sy / n)


def upward_ray_hit_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    x: float,
    y: float,
    z_low: float,
    *,
    exclude_body_id: int = -1,
) -> float | None:
    """Cast a vertical ray upward; return distance to first hit, or ``None`` if open sky (no hit)."""
    pnt = np.array([float(x), float(y), float(z_low)], dtype=np.float64)
    vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    gidbuf = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, exclude_body_id, gidbuf)
    if dist < 0.0:
        return None
    return float(dist)


def _horizontal_cardinal_ray_voids_and_finite(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    x: float,
    y: float,
    z_probe: float,
    *,
    exclude_body_id: int = -1,
) -> tuple[int, list[float]]:
    """Cardinal +x,-x,+y,-y ``mj_ray`` results: void count and list of finite hit distances."""
    pz = float(z_probe)
    gidbuf = np.zeros(1, dtype=np.int32)
    voids = 0
    finite: list[float] = []
    for dx, dy in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
        pnt = np.array([float(x), float(y), pz], dtype=np.float64)
        vec = np.array([dx, dy, 0.0], dtype=np.float64)
        dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, exclude_body_id, gidbuf)
        if dist < 0.0:
            voids += 1
        else:
            finite.append(float(dist))
    return voids, finite


def horizontal_spawn_rejects_exterior_tongue(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    x: float,
    y: float,
    z_probe: float,
    *,
    exclude_body_id: int = -1,
) -> bool:
    """True if horizontal cardinals suggest infinite-floor **exterior** (skip this XY).

    Uses cardinal ``mj_ray`` distances (excluding the robot subtree). Interior rooms usually have
    several multi-meter hits; house-edge tongues often show two or more voids **and** every finite
    hit is very short.
    """
    voids, finite = _horizontal_cardinal_ray_voids_and_finite(
        model, data, x, y, z_probe, exclude_body_id=exclude_body_id
    )
    if voids >= 3:
        return True
    if voids >= 2:
        if not finite:
            return True
        return max(finite) < float(_HORIZ_EXTERIOR_MAX_FINITE_M)
    return False


def molmospaces_placed_pose_passes_horizontal_interior_gate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    placed: tuple[float, float, float],
    floor_geom_name: str = "floor",
    require_upward_ceiling_hit: bool = False,
    min_upward_clearance_m: float | None = None,
) -> bool:
    """True if the placed base (x, y) passes post-spawn checks (horizontal tongue by default).

    Recomputes walkable floor *z* under the placed XY and runs the same horizontal cardinal ``mj_ray``
    **exterior-tongue** rule used during spawn. Call immediately after :func:`find_molmospaces_freejoint_xyz`
    (with ``data`` already at the returned pose and ``mj_forward`` applied).

    **What this does *not* assert by default:** an upward ``mj_ray`` hit (ceiling). Many iTHOR merges
    have no ceiling geom; spawn allows **open sky** (no upward hit) in that case. Set
    ``require_upward_ceiling_hit=True`` for scenes where you expect real overhead geometry and want
    to assert ``+z`` clearance (e.g. closed rooms in synthetic tests like ``MEGA_SHELL_OFFSET``).
    Use *min_upward_clearance_m* to tune the threshold when the floor probe sits just under a low
    deck (defaults to ``_MIN_UPWARD_CEILING_CLEARANCE_M`` when omitted).
    """
    x, y = float(placed[0]), float(placed[1])
    floor_eff = effective_floor_geom_name(model, floor_geom_name)
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    excl = int(base_bid) if base_bid >= 0 else -1
    mujoco.mj_forward(model, data)
    zf = walkable_floor_z_at_xy(model, data, x, y, floor_geom_name=floor_eff, exclude_body_id=excl)
    if zf is None:
        return False
    z_probe = float(zf) + 0.08
    if horizontal_spawn_rejects_exterior_tongue(model, data, x, y, z_probe, exclude_body_id=excl):
        return False
    if require_upward_ceiling_hit:
        min_up = (
            float(min_upward_clearance_m)
            if min_upward_clearance_m is not None
            else float(_MIN_UPWARD_CEILING_CLEARANCE_M)
        )
        up = upward_ray_hit_distance(model, data, x, y, z_probe, exclude_body_id=excl)
        if up is None or up < min_up:
            return False
    return True


def walkable_floor_z_at_xy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    x: float,
    y: float,
    *,
    floor_geom_name: str = "floor",
    z_beam_top: float = 1.55,
    step_past_hit: float = 0.06,
    max_segments: int = 80,
    exclude_body_id: int = -1,
) -> float | None:
    """Return world *z* of the walkable floor under (x, y), or None if the ray never hits *floor*.

    A single downward ray often hits ceiling or furniture first; this walks the ray in
    segments until the ``floor`` geom is hit or the beam leaves a plausible band.

    Geoms with **no collision** (``contype==0`` and ``conaffinity==0``), typical of iTHOR visual
    meshes, are stepped past with a larger advance so thin decorative shells do not exhaust
    ``max_segments`` before the global floor plane. A **floor-aligned** visual hit (``z <= 0.12``)
    is treated as walkable height: stepping past would push the probe under the infinite floor
    where the next ray often returns no hit.
    """
    floor_resolved = effective_floor_geom_name(model, floor_geom_name)
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_resolved)
    if floor_gid < 0:
        # No registered floor geom name; still ray-cast using visual-floor / plane heuristics below.
        floor_gid = -1

    pnt = np.array([float(x), float(y), float(z_beam_top)], dtype=np.float64)
    vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    for _ in range(max_segments):
        gidbuf = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, exclude_body_id, gidbuf)
        if dist < 0:
            return None
        hit_gid = int(gidbuf[0])
        if hit_gid < 0 or hit_gid >= int(model.ngeom):
            return None
        hit = pnt + float(dist) * vec
        if hit_gid == floor_gid:
            return float(hit[2])
        ct = int(model.geom_contype[hit_gid])
        ca = int(model.geom_conaffinity[hit_gid])
        # iTHOR often stacks large **visual-only** floor meshes at z≈0. Stepping past them moves the
        # probe below the infinite floor plane where the next ``mj_ray`` commonly returns no hit.
        if ct == 0 and ca == 0 and float(hit[2]) <= 0.12:
            return float(hit[2])
        step = float(step_past_hit)
        if ct == 0 and ca == 0:
            rb = float(model.geom_rbound[hit_gid])
            step = max(step, min(0.52, 2.4 * rb + 0.05))
        pnt = hit + vec * step
        if pnt[2] < -2.5 or pnt[2] > z_beam_top + 2.5:
            return None
    return None


def worst_robot_nonfloor_contact_dist(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    floor_geom_name: str = "floor",
) -> float:
    """After :func:`mujoco.mj_forward`, run collision and return the minimum ``contact.dist`` for
    any contact that involves the robot and a geom other than *floor*.

    Robot–robot pairs are ignored (internal links). Values < 0 mean penetration into scene clutter.
    """
    floor_resolved = effective_floor_geom_name(model, floor_geom_name)
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_resolved)
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    if floor_gid < 0 or base_bid < 0:
        return 0.0
    robot_bodies = _bodies_descending_from(model, base_bid)
    mujoco.mj_collision(model, data)
    worst = 1.0
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        r1 = _geom_body_is_robot(model, g1, robot_bodies)
        r2 = _geom_body_is_robot(model, g2, robot_bodies)
        if not r1 and not r2:
            continue
        if r1 and r2:
            continue
        if g1 == floor_gid or g2 == floor_gid:
            continue
        dist = float(c.dist)
        if dist < worst:
            worst = dist
    return worst


def format_spawn_contact_report(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    floor_geom_name: str = "floor",
    max_lines: int = 40,
    dist_report_threshold: float = 0.12,
) -> list[str]:
    """After :func:`mujoco.mj_forward`, run collision and list robot–scene contacts (for spawn QA)."""
    floor_eff = effective_floor_geom_name(model, floor_geom_name)
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_eff)
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    if floor_gid < 0 or base_bid < 0:
        return [f"contact_report: missing floor ({floor_eff!r}) or base body {base_body_name!r}"]
    robot_bodies = _bodies_descending_from(model, base_bid)
    mujoco.mj_collision(model, data)
    lines: list[str] = [
        f"contact_report: ncon={data.ncon} floor_geom={floor_eff!r} worst_nonfloor="
        f"{worst_robot_nonfloor_contact_dist(model, data, base_body_name=base_body_name, floor_geom_name=floor_geom_name):.4f}"
    ]
    n = 0
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        r1 = _geom_body_is_robot(model, g1, robot_bodies)
        r2 = _geom_body_is_robot(model, g2, robot_bodies)
        if not r1 and not r2:
            continue
        if r1 and r2:
            continue
        if g1 == floor_gid or g2 == floor_gid:
            continue
        dist = float(c.dist)
        if dist > dist_report_threshold:
            continue
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom{g1}"
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom{g2}"
        lines.append(f"  con[{i}] dist={dist:+.5f} {n1!r} <-> {n2!r}")
        n += 1
        if n >= max_lines:
            lines.append(f"  ... truncated after {max_lines} scene contacts")
            break
    if n == 0:
        lines.append("  (no robot–scene contacts under dist threshold besides floor)")
    return lines


def format_spawn_floor_alignment_report(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    floor_geom_name: str = "floor",
    xy: tuple[float, float],
) -> list[str]:
    """One-line summary: floor height under *xy* vs lowest robot collision geom (after ``mj_forward``)."""
    floor_eff = effective_floor_geom_name(model, floor_geom_name)
    x, y = float(xy[0]), float(xy[1])
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    if base_bid < 0:
        return [f"floor_align: missing base body {base_body_name!r}"]
    robot_bodies = _bodies_descending_from(model, base_bid)
    ray_excl = int(base_bid)
    mujoco.mj_forward(model, data)
    zf = walkable_floor_z_at_xy(model, data, x, y, floor_geom_name=floor_eff, exclude_body_id=ray_excl)
    from emet.simulation.spawn_settle import _min_robot_collision_geom_bottom_z

    zb = _min_robot_collision_geom_bottom_z(model, data, robot_bodies)
    worst = worst_robot_nonfloor_contact_dist(model, data, base_body_name=base_body_name, floor_geom_name=floor_eff)
    zf_s = f"{zf:.4f}" if zf is not None else "None"
    zb_s = f"{zb:.4f}" if zb is not None else "None"
    delta_s = "n/a"
    if zf is not None and zb is not None:
        delta_s = f"{float(zb) - float(zf):+.4f}"
    return [
        f"floor_align: floor_geom={floor_eff!r} z_floor_under_xy={zf_s} zb_geom_min={zb_s} "
        f"zb_minus_zfloor={delta_s} worst_nonfloor={worst:.5f}"
    ]


def iter_annulus_xy_candidates(
    r_min: float = 0.35,
    r_max: float = 3.2,
    *,
    n_radii: int = 22,
    base_angles_per_ring: int = 12,
    xy_clip: tuple[float, float, float, float] | None = None,
    xy_origin: tuple[float, float] = (0.0, 0.0),
) -> Iterable[tuple[float, float]]:
    """Polar grid from *r_min* to *r_max* around *xy_origin* (inner rings first).

    iTHOR / Molmo scenes are often **not** centered on the world origin; sampling only around
    ``(0, 0)`` misses the house footprint and yields spawns on open floor outside walls.

    If *xy_clip* is ``(xmin, xmax, ymin, ymax)``, only yields points inside that rectangle.
    """
    ox, oy = float(xy_origin[0]), float(xy_origin[1])
    radii = np.linspace(r_min, r_max, n_radii)
    for r in radii:
        n_ang = max(base_angles_per_ring, int(base_angles_per_ring + 8 * (r / r_max)))
        for k in range(n_ang):
            th = (2.0 * math.pi * k) / n_ang
            x = ox + float(r * math.cos(th))
            y = oy + float(r * math.sin(th))
            if xy_clip is not None:
                cx0, cx1, cy0, cy1 = xy_clip
                if not (cx0 <= x <= cx1 and cy0 <= y <= cy1):
                    continue
            yield x, y


def _coarse_grid_xy_in_clip(
    xy_clip: tuple[float, float, float, float],
    *,
    step: float = 0.62,
    max_points: int = 220,
) -> list[tuple[float, float]]:
    """Regular grid inside *xy_clip*, center-sorted (interior-first for typical houses)."""
    x0, x1, y0, y1 = xy_clip
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    pts: list[tuple[float, float]] = []
    xs = np.arange(x0 + step * 0.2, x1, step)
    ys = np.arange(y0 + step * 0.2, y1, step)
    for px in xs:
        for py in ys:
            pts.append((float(px), float(py)))
    pts.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
    if len(pts) > max_points:
        pts = pts[:max_points]
    return pts
