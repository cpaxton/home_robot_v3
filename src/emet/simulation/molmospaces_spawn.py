# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source code tree.

"""Spawn placement for MolmoSpaces (scene MJCF + merged mobile robot).

Merged scenes place the robot at the world origin on a ``freejoint``. The global ``floor`` plane
extends beyond walls, so XY must stay inside the **collision footprint** of room content and
points where an **upward** ray hits geometry (open void returns no hit). We then avoid furniture
penetration using contact scoring and optional vertical refinement.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from typing import Any

import mujoco
import numpy as np

import emet.utils.logger as log

# Reject XY where an upward ray finds nothing (open void) or only a glancing hit very close above
# the probe (likely under a thin shelf / numerical noise).
_MIN_UPWARD_CEILING_CLEARANCE_M = 0.15

logger = log.Logger(__name__)


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
    return None


def effective_floor_geom_name(model: mujoco.MjModel, floor_geom_name: str = "floor") -> str:
    """Return *floor_geom_name* if present, else :func:`resolve_floor_geom_name` or the hint string."""
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_geom_name) >= 0:
        return floor_geom_name
    resolved = resolve_floor_geom_name(model)
    return resolved if resolved is not None else floor_geom_name


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
        if rb <= 0.0 or rb > max_geom_rbound:
            continue
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


def _erode_xy_rect(
    rect: tuple[float, float, float, float], inset: float
) -> tuple[float, float, float, float] | None:
    """Shrink a clip rectangle symmetrically; return ``None`` if it would collapse."""
    x0, x1, y0, y1 = rect
    if x1 - x0 < 2.0 * inset + 0.9 or y1 - y0 < 2.0 * inset + 0.9:
        return None
    return (x0 + inset, x1 - inset, y0 + inset, y1 - inset)


def scene_collision_centroid_xy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    robot_bodies: set[int],
    floor_geom_name: str = "floor",
    max_geom_rbound: float = 15.0,
    suppress_exterior_filter: bool = False,
) -> tuple[float, float] | None:
    """Mean (x, y) of scene collision geoms (same filters as :func:`collision_scene_xy_clip_rect`)."""
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
        if rb <= 0.0 or rb > max_geom_rbound:
            continue
        p = data.geom_xpos[g]
        sx += float(p[0])
        sy += float(p[1])
        n += 1
    if n < 8:
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


def walkable_floor_z_at_xy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    x: float,
    y: float,
    *,
    floor_geom_name: str = "floor",
    z_beam_top: float = 1.55,
    step_past_hit: float = 0.06,
    max_segments: int = 48,
    exclude_body_id: int = -1,
) -> float | None:
    """Return world *z* of the walkable floor under (x, y), or None if the ray never hits *floor*.

    A single downward ray often hits ceiling or furniture first; this walks the ray in
    segments until the ``floor`` geom is hit or the beam leaves a plausible band.
    """
    floor_resolved = effective_floor_geom_name(model, floor_geom_name)
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_resolved)
    if floor_gid < 0:
        return None
    pnt = np.array([float(x), float(y), float(z_beam_top)], dtype=np.float64)
    vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    for _ in range(max_segments):
        gidbuf = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, exclude_body_id, gidbuf)
        if dist < 0:
            return None
        hit_gid = int(gidbuf[0])
        hit = pnt + float(dist) * vec
        if hit_gid == floor_gid:
            return float(hit[2])
        pnt = hit + vec * float(step_past_hit)
        if pnt[2] < -1.5 or pnt[2] > z_beam_top + 2.0:
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
            return None
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
    max_steps: int = 160,
    step_m: float = 0.0028,
    target_foot_clearance_above_floor_m: float = 0.018,
    min_z_above_floor_m: float = -0.02,
) -> float:
    """Lower base *z* until collision hull is near *z_floor* while keeping non-floor contacts acceptable."""
    z = float(z_start)
    best = z
    for _ in range(max_steps):
        if not write_freejoint_base_xyzw(
            model, data, base_body_name=base_body_name, x=x, y=y, z=z
        ):
            break
        mujoco.mj_forward(model, data)
        worst = worst_robot_nonfloor_contact_dist(
            model, data, base_body_name=base_body_name, floor_geom_name=floor_geom_name
        )
        if worst < min_nonfloor_clearance - 1e-6:
            break
        zb = _min_robot_collision_geom_bottom_z(model, data, robot_bodies)
        best = z
        if zb is None:
            break
        if zb <= z_floor + target_foot_clearance_above_floor_m + 2e-4:
            break
        z -= step_m
        if z < z_floor + min_z_above_floor_m:
            break
    write_freejoint_base_xyzw(model, data, base_body_name=base_body_name, x=x, y=y, z=best)
    mujoco.mj_forward(model, data)
    return best


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

    r_annulus_max = 3.2
    if xy_clip is not None:
        x0c, x1c, y0c, y1c = xy_clip
        half_diag = 0.5 * math.hypot(x1c - x0c, y1c - y0c)
        r_annulus_max = float(min(14.5, max(3.8, 0.52 * half_diag + 0.65)))

    candidates = list(
        iter_annulus_xy_candidates(
            r_max=r_annulus_max,
            xy_clip=xy_clip,
            xy_origin=(ox, oy),
        )
    )
    if xy_clip is not None and len(candidates) < 72:
        seen = {(round(a, 2), round(b, 2)) for a, b in candidates}
        for px, py in _coarse_grid_xy_in_clip(xy_clip, step=0.62, max_points=220):
            key = (round(px, 2), round(py, 2))
            if key in seen:
                continue
            seen.add(key)
            candidates.append((px, py))

    if centroid is not None:
        cx, cy = centroid
        candidates.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
    elif xy_clip is not None:
        cx = 0.5 * (xy_clip[0] + xy_clip[1])
        cy = 0.5 * (xy_clip[2] + xy_clip[3])
        candidates.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)

    best_xy: tuple[float, float, float, float] | None = None  # x, y, z_floor, worst_score
    for x, y in candidates:
        z_floor = walkable_floor_z_at_xy(
            model, data, x, y, floor_geom_name=floor_effective, exclude_body_id=ray_exclude
        )
        if z_floor is None:
            continue
        z_probe = float(z_floor) + 0.08
        up_dist = upward_ray_hit_distance(
            model, data, x, y, z_probe, exclude_body_id=ray_exclude
        )
        if up_dist is None or up_dist < min_upward_clearance:
            continue
        for zm in z_margins:
            z = z_floor + float(zm)
            if not write_freejoint_base_xyzw(model, data, base_body_name=base_body_name, x=x, y=y, z=z):
                return None
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
                )
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
        )
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
) -> tuple[float, float, float] | None:
    """Single (x,y): walkable floor, upward clearance, z margin sweep + settle (shared by fallbacks)."""
    z_floor = walkable_floor_z_at_xy(
        model, data, x, y, floor_geom_name=floor_effective, exclude_body_id=ray_exclude
    )
    if z_floor is None:
        return None
    z_probe = float(z_floor) + 0.08
    up_dist = upward_ray_hit_distance(model, data, x, y, z_probe, exclude_body_id=ray_exclude)
    if up_dist is not None and up_dist < min_upward_clearance:
        return None
    for zm in z_margins:
        z = z_floor + float(zm)
        if not write_freejoint_base_xyzw(model, data, base_body_name=base_body_name, x=x, y=y, z=z):
            return None
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
            )
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
) -> tuple[float, float, float] | None:
    """Last resort: a few XY points near the collision clip center with looser ceiling / contact."""
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
) -> tuple[float, float, float] | None:
    """Return a world-space base position (x, y, z) for the robot free joint, or None to keep defaults.

    Requires **no meaningful penetration** (``contact.dist >= min_nonfloor_clearance``) vs scene
    geoms other than *floor* (tiny negative values are treated as numerical noise).
    Uses a coarse polar XY search (clipped to scene collision footprint when available), rejects
    points under open sky (no upward ray hit), then a finer vertical sweep on the best XY when needed.
    Leaves *data* at the chosen pose with :func:`mujoco.mj_forward` applied.
    """
    floor_effective = effective_floor_geom_name(model, floor_geom_name)
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    robot_bodies: set[int] = set()
    if base_bid >= 0:
        robot_bodies = _bodies_descending_from(model, base_bid)

    clip_probe = collision_scene_xy_clip_rect(
        model, data, robot_bodies=robot_bodies, floor_geom_name=floor_effective, max_geom_rbound=15.0
    )

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
    )
    if placed is not None:
        return placed

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
    )
    if placed_relaxed is not None:
        return placed_relaxed

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
    )
    if placed_fb is not None:
        return placed_fb

    label = scene_label or "(unknown scene)"
    clip_s = "yes" if clip_probe is not None else "no"
    logger.warning(
        f"molmospaces_spawn.find_molmospaces_freejoint_xyz: primary, relaxed, and clip-center "
        f"fallback failed (scene={label!r} floor_geom_resolved={floor_effective!r} xy_clip_rect={clip_s})"
    )
    return None

