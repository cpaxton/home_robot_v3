# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source code tree.

"""Spawn placement for MolmoSpaces (scene MJCF + merged mobile robot).

Merged scenes place the robot at the world origin on a ``freejoint``. The global ``floor`` plane
extends beyond walls, so XY must stay inside the **collision footprint** of room content and
points where an **upward** ray, when present, is not too close to overhead geometry (a very near
hit still rejects, as under a shelf). **Open sky** (no upward hit) is allowed because many iTHOR
merged scenes omit an explicit ceiling while ``walkable_floor_z_at_xy`` still finds the floor.
We also skip XY where
**horizontal** cardinal ``mj_ray`` hits suggest an infinite-floor **exterior** tongue (several
open directions with only very short finite hits). If no valid pose is found, placement returns
``None`` and the caller must not move the base to an invalid exterior tongue.
We then avoid furniture penetration using contact scoring and optional vertical refinement.

The upstream ``molmo_spaces`` Python package (scene lists / resource install) does not define a
merged-robot spawn policy; emet autoplace owns XY/Z on the free joint after MJCF merge.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np

import emet.utils.logger as log
from emet.robots.base import RobotSpec

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

logger = log.Logger(__name__)


def spawn_debug_enabled() -> bool:
    """True when ``EMET_MOLMOSPACES_SPAWN_DEBUG`` is set or ``--debug-molmospaces-spawn`` turned it on.

    When true, :func:`find_molmospaces_freejoint_xyz` logs a downsampled ASCII occupancy map and
    writes a PNG: use ``EMET_MOLMOSPACES_SPAWN_DEBUG_MAP_PNG`` for an explicit path, or leave it unset
    to default to ``./molmospaces_spawn_topdown.png`` in the process cwd. Set
    ``EMET_MOLMOSPACES_SPAWN_DEBUG_MAP_PNG=0`` to skip the default PNG (ASCII still logs).

    ASCII legend (see ``topdown_map_key`` log line): ``'.'`` free, ``'#'`` blocked, ``'='`` collision
    clip, ``'*'`` first 72 occupancy-priority samples, ``'o'`` base before autoplace, ``'@'`` chosen
    spawn (robot base XY).
    """
    v = os.environ.get("EMET_MOLMOSPACES_SPAWN_DEBUG", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def spawn_dbg(msg: str) -> None:
    if spawn_debug_enabled():
        logger.info(f"[molmospaces_spawn] {msg}")


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
    raw = (
        robocasa_autoplace_env
        if robocasa_autoplace_env is not None
        else os.environ.get("EMET_ROBOSUITE_AUTOPLACE", "1")
    )
    v = str(raw).strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True


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


def _erode_xy_rect(
    rect: tuple[float, float, float, float], inset: float
) -> tuple[float, float, float, float] | None:
    """Shrink a clip rectangle symmetrically; return ``None`` if it would collapse."""
    x0, x1, y0, y1 = rect
    if x1 - x0 < 2.0 * inset + 0.9 or y1 - y0 < 2.0 * inset + 0.9:
        return None
    return (x0 + inset, x1 - inset, y0 + inset, y1 - inset)


def _clamp_xy_into_rect(x: float, y: float, rect: tuple[float, float, float, float]) -> tuple[float, float]:
    """Clamp *x*, *y* to the closed axis-aligned rectangle (xmin, xmax, ymin, ymax)."""
    x0, x1, y0, y1 = rect
    return (min(max(float(x), x0), x1), min(max(float(y), y0), y1))


def _max_xy_distance_to_rect_corners(
    x: float, y: float, rect: tuple[float, float, float, float]
) -> float:
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
    zf = walkable_floor_z_at_xy(
        model, data, x, y, floor_geom_name=floor_eff, exclude_body_id=excl
    )
    if zf is None:
        return False
    z_probe = float(zf) + 0.08
    if horizontal_spawn_rejects_exterior_tongue(
        model, data, x, y, z_probe, exclude_body_id=excl
    ):
        return False
    if require_upward_ceiling_hit:
        min_up = (
            float(min_upward_clearance_m)
            if min_upward_clearance_m is not None
            else float(_MIN_UPWARD_CEILING_CLEARANCE_M)
        )
        up = upward_ray_hit_distance(
            model, data, x, y, z_probe, exclude_body_id=excl
        )
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
    zf = walkable_floor_z_at_xy(
        model, data, x, y, floor_geom_name=floor_eff, exclude_body_id=ray_excl
    )
    zb = _min_robot_collision_geom_bottom_z(model, data, robot_bodies)
    worst = worst_robot_nonfloor_contact_dist(
        model, data, base_body_name=base_body_name, floor_geom_name=floor_eff
    )
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
    max_steps: int = 320,
    step_m: float = 0.0028,
    target_foot_clearance_above_floor_m: float = 0.018,
    min_z_above_floor_m: float = -0.02,
) -> float | None:
    """Lower base *z* until collision hull is near *z_floor* while keeping non-floor contacts acceptable.

    Returns ``None`` if no pose along the descent had acceptable non-floor clearance (e.g. base
    starts intersecting a wall so the first ``mj_forward`` already fails the clearance test).
    """
    z = float(z_start)
    last_good_z: float | None = None
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
        last_good_z = z
        zb = _min_robot_collision_geom_bottom_z(model, data, robot_bodies)
        if zb is None:
            break
        if zb <= z_floor + target_foot_clearance_above_floor_m + 2e-4:
            break
        z -= step_m
        if z < z_floor + min_z_above_floor_m:
            break
    if last_good_z is None:
        return None
    write_freejoint_base_xyzw(
        model, data, base_body_name=base_body_name, x=x, y=y, z=float(last_good_z)
    )
    mujoco.mj_forward(model, data)
    return float(last_good_z)


def _post_settle_pose_acceptable(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    floor_geom_name: str,
    z_floor: float,
    robot_bodies: set[int],
    min_nonfloor_clearance: float,
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
    zb = _min_robot_collision_geom_bottom_z(model, data, robot_bodies)
    if zb is None:
        return False
    if zb < z_floor - max_foot_below_floor_m:
        return False
    if zb > z_floor + max_foot_above_floor_m:
        return False
    return True


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
            merged_mjcf_path, camera=None, agent_radius=agent_radius, px_per_m=px_per_m
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


def _spawn_debug_downsample_occ(occ: np.ndarray, max_h: int, max_w: int) -> tuple[np.ndarray, int, int]:
    """Return ``(nh, nw)`` uint8 grid ``1`` = free, ``0`` = blocked, plus strides ``sh, sw``."""
    H, W = int(occ.shape[0]), int(occ.shape[1])
    sh = max(1, (H + max_h - 1) // max_h)
    sw = max(1, (W + max_w - 1) // max_w)
    nh = (H + sh - 1) // sh
    nw = (W + sw - 1) // sw
    out = np.zeros((nh, nw), dtype=np.uint8)
    for i in range(nh):
        r0, r1 = i * sh, min(H, (i + 1) * sh)
        for j in range(nw):
            c0, c1 = j * sw, min(W, (j + 1) * sw)
            patch = occ[r0:r1, c0:c1]
            if patch.size == 0:
                continue
            out[i, j] = 1 if float(np.mean(patch.astype(np.float64))) > 0.5 else 0
    return out, sh, sw


def _spawn_debug_ascii_topdown(
    ithor_map: Any,
    *,
    clip_probe: tuple[float, float, float, float] | None,
    occ_priority_xy: list[tuple[float, float]],
    placed: tuple[float, float, float] | None,
    initial_xy: tuple[float, float] | None,
) -> list[str]:
    """Build printable ASCII lines (legend + map). ``ithor_map.occupancy`` is ``True`` = free."""
    max_h, max_w = 52, 88
    occ = np.asarray(ithor_map.occupancy, dtype=bool)
    H, W = occ.shape
    small, sh, sw = _spawn_debug_downsample_occ(occ, max_h, max_w)
    nh, nw = small.shape
    ch: list[list[str]] = [["#" if small[i, j] == 0 else "." for j in range(nw)] for i in range(nh)]

    def world_to_small(x: float, y: float) -> tuple[int, int] | None:
        rc = _world_xy_to_occ_cell(ithor_map, x, y)
        if rc is None:
            return None
        r0, c0 = rc
        return (r0 // sh, c0 // sw)

    if clip_probe is not None:
        x0, x1, y0, y1 = clip_probe
        step = max(0.18, min(0.45, 0.02 * min(x1 - x0, y1 - y0)))
        xs = np.arange(x0, x1 + 1e-9, step, dtype=np.float64)
        ys = np.arange(y0, y1 + 1e-9, step, dtype=np.float64)
        for x in xs:
            for y in (y0, y1):
                p = world_to_small(float(x), float(y))
                if p:
                    br, bc = p
                    ch[br][bc] = "="
        for y in ys:
            for x in (x0, x1):
                p = world_to_small(float(x), float(y))
                if p:
                    br, bc = p
                    ch[br][bc] = "="

    for px, py in occ_priority_xy[:72]:
        p = world_to_small(px, py)
        if p:
            br, bc = p
            if ch[br][bc] in (".", "="):
                ch[br][bc] = "*"

    if initial_xy is not None:
        p = world_to_small(initial_xy[0], initial_xy[1])
        if p:
            br, bc = p
            if ch[br][bc] != "@":
                ch[br][bc] = "o"

    if placed is not None:
        p = world_to_small(placed[0], placed[1])
        if p:
            br, bc = p
            ch[br][bc] = "@"

    lines = [
        "topdown_map_key: '.'=occupancy_free '#'=blocked '='=collision_clip_rect "
        "'*'=occ_priority_xy[:72]_on_free 'o'=base_xy_before_autoplace '@'=chosen_spawn_base_xy",
        "topdown_map: occupancy downsampled "
        f"(orig {H}x{W} stride {sh}x{sw} -> {nh}x{nw}; same symbol key as topdown_map_key line above)",
    ]
    for i in range(nh):
        lines.append("topdown_map: " + "".join(ch[i][j] for j in range(nw)))
    return lines


def _spawn_debug_write_occupancy_png(
    ithor_map: Any,
    path: str,
    *,
    clip_probe: tuple[float, float, float, float] | None,
    occ_priority_xy: list[tuple[float, float]],
    placed: tuple[float, float, float] | None,
    initial_xy: tuple[float, float] | None,
) -> None:
    from PIL import Image, ImageDraw

    occ = np.asarray(ithor_map.occupancy, dtype=np.uint8)
    rgb = np.stack([255 - occ * 255] * 3, axis=-1)
    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)
    H, W = occ.shape

    def wc(x: float, y: float) -> tuple[int, int] | None:
        rc = _world_xy_to_occ_cell(ithor_map, x, y)
        if rc is None:
            return None
        row, col = int(rc[0]), int(rc[1])
        return (col, row)  # PIL (x=col, y=row)

    if clip_probe is not None:
        x0, x1, y0, y1 = clip_probe
        poly = []
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)):
            p = wc(x, y)
            if p:
                poly.append(p)
        if len(poly) >= 2:
            draw.line(poly, fill=(255, 80, 80), width=2)
    for px, py in occ_priority_xy[:200]:
        p = wc(px, py)
        if p:
            draw.rectangle((p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1), fill=(80, 200, 255))
    if initial_xy is not None:
        p = wc(initial_xy[0], initial_xy[1])
        if p:
            draw.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), outline=(255, 200, 0), width=2)
    if placed is not None:
        p = wc(placed[0], placed[1])
        if p:
            draw.ellipse((p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5), outline=(50, 255, 80), width=2)
    outp = Path(path).expanduser()
    outp.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(outp))


def _spawn_debug_emit_topdown(
    ithor_map: Any | None,
    *,
    clip_probe: tuple[float, float, float, float] | None,
    occ_priority_xy: list[tuple[float, float]],
    placed: tuple[float, float, float] | None,
    how: str,
    initial_xy: tuple[float, float] | None,
) -> None:
    if not spawn_debug_enabled():
        return
    spawn_dbg(
        f"spawn_debug_summary: how={how!r} placed={placed!r} initial_base_xy={initial_xy!r} "
        f"n_occ_priority={len(occ_priority_xy)} clip_probe={'set' if clip_probe is not None else 'none'}"
    )
    if ithor_map is None:
        spawn_dbg("topdown_map: (no iTHOR occupancy — enable EMET_MOLMOSPACES_OCC_MAP or use iTHOR scene)")
        return
    try:
        for line in _spawn_debug_ascii_topdown(
            ithor_map,
            clip_probe=clip_probe,
            occ_priority_xy=occ_priority_xy,
            placed=placed,
            initial_xy=initial_xy,
        ):
            spawn_dbg(line)
    except Exception as e:
        spawn_dbg(f"topdown_map_ascii: failed ({e!r})")

    png_raw = os.environ.get("EMET_MOLMOSPACES_SPAWN_DEBUG_MAP_PNG", "")
    png_path = png_raw.strip()
    if png_path.lower() in ("0", "false", "no", "none", "off"):
        png_path = ""
    elif not png_path:
        png_path = str(Path.cwd() / "molmospaces_spawn_topdown.png")
    if png_path:
        try:
            _spawn_debug_write_occupancy_png(
                ithor_map,
                png_path,
                clip_probe=clip_probe,
                occ_priority_xy=occ_priority_xy,
                placed=placed,
                initial_xy=initial_xy,
            )
            spawn_dbg(f"topdown_map_png: wrote {png_path!r}")
        except Exception as e:
            spawn_dbg(f"topdown_map_png: failed ({e!r})")


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
        up_dist = upward_ray_hit_distance(
            model, data, x, y, z_probe, exclude_body_id=ray_exclude
        )
        # Open sky (no upward hit) is common in iTHOR MJCF without an explicit ceiling geom; do not
        # treat that like "void under a shelf". Reject only a *close* ceiling when a hit exists.
        if up_dist is not None and up_dist < min_upward_clearance:
            continue
        if horizontal_spawn_rejects_exterior_tongue(
            model, data, x, y, z_probe, exclude_body_id=ray_exclude
        ):
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
) -> tuple[float, float, float] | None:
    """Single (x,y): walkable floor, upward clearance, z margin sweep + settle (shared by fallbacks)."""
    z_air = max(8.0, 3.0 * float(model.stat.extent))
    if not write_freejoint_base_xyzw(
        model, data, base_body_name=base_body_name, x=float(x), y=float(y), z=z_air
    ):
        return None
    mujoco.mj_forward(model, data)
    z_floor = walkable_floor_z_at_xy(
        model, data, x, y, floor_geom_name=floor_effective, exclude_body_id=ray_exclude
    )
    if z_floor is None:
        return None
    z_probe = float(z_floor) + 0.08
    up_dist = upward_ray_hit_distance(model, data, x, y, z_probe, exclude_body_id=ray_exclude)
    if up_dist is not None and up_dist < min_upward_clearance:
        return None
    if horizontal_spawn_rejects_exterior_tongue(
        model, data, x, y, z_probe, exclude_body_id=ray_exclude
    ):
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
) -> tuple[float, float, float] | None:
    """Return a world-space base position (x, y, z) for the robot free joint, or None to keep defaults.

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

    occ_xy, occ_map_dbg = _ithor_occupancy_priority_xy(merged_mjcf_path, environment)

    def _spawn_debug_finish(
        out: tuple[float, float, float] | None, how: str
    ) -> tuple[float, float, float] | None:
        _spawn_debug_emit_topdown(
            occ_map_dbg,
            clip_probe=clip_probe,
            occ_priority_xy=occ_xy,
            placed=out,
            how=how,
            initial_xy=initial_base_xy,
        )
        return out

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


def infer_planar_anchor_body_name(
    model: mujoco.MjModel, joint_names: tuple[str, str, str]
) -> str | None:
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
    mujoco.mj_forward(model, data)
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
    mujoco.mj_forward(model, data)
    return True


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
) -> tuple[float, float, float] | None:
    """Search collision-free world `(x, y, yaw)` for planar slide+yaw bases (Robocasa + Maurice-style MJCF).

    Uses the same collision clip / floor probe / contact scoring ideas as free-joint Molmo spawn,
    but applies poses via :func:`write_planar_base_xyt`. For ``spawn_profile == \"robocasa\"`` we
    disable upward-ray rejection (open ceilings / cabinet tops) but **keep** the horizontal
    exterior-tongue filter so we do not pick the infinite floor outside the kitchen footprint.
    ``spawn_hint_xyt`` is accepted for API compatibility but **not** used for ordering (Robosuite
    placeholder base pose can disagree with the merged MJCF frame).

    Candidate *(x, y)* and *yaw* are **world** / kitchen-map coordinates; they are mapped to slide
    joint values using the planar anchor body (parent of the first slide), e.g. after Robocasa
    strip-and-replace when ``base_root`` carries a non-identity ``pos`` / ``quat``.

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

    occ_xy, _ = _ithor_occupancy_priority_xy(merged_mjcf_path, environment)
    priority_xy: list[tuple[float, float]] = []
    if occ_xy:
        seen_xy = {(round(a, 2), round(b, 2)) for a, b in base_candidates}
        for px, py in occ_xy:
            k = (round(px, 2), round(py, 2))
            if k in seen_xy:
                continue
            seen_xy.add(k)
            priority_xy.append((float(px), float(py)))

    candidates = priority_xy + base_candidates
    if xy_clip is not None:
        cx = 0.5 * (xy_clip[0] + xy_clip[1])
        cy = 0.5 * (xy_clip[2] + xy_clip[3])
        candidates.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
    elif centroid is not None:
        cx, cy = float(centroid[0]), float(centroid[1])
        candidates.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)

    yaws = [float(k * math.pi / 4.0) for k in range(8)]
    min_upward = -1.0 if spawn_profile == "robocasa" else float(_MIN_UPWARD_CEILING_CLEARANCE_M)
    clearance_passes = (
        (0.045, "clear045"),
        (0.028, "clear028"),
        (0.014, "clear014"),
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
                    up_dist = upward_ray_hit_distance(
                        model, data, x, y, z_probe, exclude_body_id=ray_exclude
                    )
                    if up_dist is not None and up_dist < min_upward:
                        continue
                if horizontal_spawn_rejects_exterior_tongue(
                    model, data, x, y, z_probe, exclude_body_id=ray_exclude
                ):
                    continue
                worst = worst_robot_nonfloor_contact_dist(
                    model, data, base_body_name=base_body_name, floor_geom_name=floor_effective
                )
                if worst >= float(min_clear):
                    if xy_clip_scene is not None and base_bid >= 0:
                        bx = float(data.body(base_body_name).xpos[0])
                        by = float(data.body(base_body_name).xpos[1])
                        x0s, x1s, y0s, y1s = xy_clip_scene
                        pad = 0.22
                        if not (x0s + pad <= bx <= x1s - pad and y0s + pad <= by <= y1s - pad):
                            continue
                    spawn_dbg(f"planar_find: OK pass={tag!r} xy=({x:.3f},{y:.3f}) yaw={yaw:.3f} worst={worst:.5f}")
                    return (float(x), float(y), float(yaw))
    spawn_dbg(f"planar_find: failed scene={scene_label!r} profile={spawn_profile!r}")
    return None
