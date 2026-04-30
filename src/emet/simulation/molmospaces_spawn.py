# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source code tree.

"""Spawn placement for MolmoSpaces (scene MJCF + merged mobile robot).

Merged scenes ship the robot at the world origin on a ``freejoint``. iTHOR-style kitchens
often have dense collision (island, counters) near (0, 0), so the base can start inside a
table and then tunnel through geometry. We pick a walkable pose by raycasting to the named
floor geom and scoring robot-vs-scene contacts (excluding floor).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import mujoco
import numpy as np


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
) -> float | None:
    """Return world *z* of the walkable floor under (x, y), or None if the ray never hits *floor*.

    A single downward ray often hits ceiling or furniture first; this walks the ray in
    segments until the ``floor`` geom is hit or the beam leaves a plausible band.
    """
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_geom_name)
    if floor_gid < 0:
        return None
    pnt = np.array([float(x), float(y), float(z_beam_top)], dtype=np.float64)
    vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    for _ in range(max_segments):
        gidbuf = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, -1, gidbuf)
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

    Values << 0 indicate deep interpenetration with furniture or walls.
    """
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_geom_name)
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
        if g1 == floor_gid or g2 == floor_gid:
            continue
        dist = float(c.dist)
        if dist < worst:
            worst = dist
    return worst


def iter_annulus_xy_candidates(
    r_min: float = 0.35,
    r_max: float = 4.0,
    *,
    n_radii: int = 18,
    base_angles_per_ring: int = 10,
) -> Iterable[tuple[float, float]]:
    """Polar grid from *r_min* to *r_max* (outer rings first — tends to clear kitchen islands)."""
    radii = np.linspace(r_max, r_min, n_radii)
    for r in radii:
        n_ang = max(base_angles_per_ring, int(base_angles_per_ring + 6 * (r / r_max)))
        for k in range(n_ang):
            th = (2.0 * math.pi * k) / n_ang
            yield float(r * math.cos(th)), float(r * math.sin(th))


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


def find_molmospaces_freejoint_xyz(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_body_name: str,
    floor_geom_name: str = "floor",
    z_margins: tuple[float, ...] = (0.12, 0.18, 0.24, 0.32, 0.40),
    penetration_ok_if_above: float = -0.008,
) -> tuple[float, float, float] | None:
    """Return a world-space base position (x, y, z) for the robot free joint, or None to keep defaults.

    Chooses the candidate with the least interpenetration (largest worst contact dist among
    robot–non-floor pairs). Leaves *data* at the chosen pose with :func:`mujoco.mj_forward` applied.
    Requires an existing :func:`mujoco.mj_forward` on *data* for ray tests.
    """
    best: tuple[float, float, float, float] | None = None  # x,y,z,score (score = worst dist)
    for x, y in iter_annulus_xy_candidates():
        z_floor = walkable_floor_z_at_xy(model, data, x, y, floor_geom_name=floor_geom_name)
        if z_floor is None:
            continue
        for zm in z_margins:
            z = z_floor + float(zm)
            if not write_freejoint_base_xyzw(model, data, base_body_name=base_body_name, x=x, y=y, z=z):
                return None
            mujoco.mj_forward(model, data)
            worst = worst_robot_nonfloor_contact_dist(
                model, data, base_body_name=base_body_name, floor_geom_name=floor_geom_name
            )
            if worst >= penetration_ok_if_above:
                return (x, y, z)
            if best is None or worst > best[3]:
                best = (x, y, z, worst)
    if best is None:
        return None
    # Use least-bad pose if nothing fully cleared the threshold (narrow gaps / mesh quirks).
    x, y, z, _ = best
    write_freejoint_base_xyzw(model, data, base_body_name=base_body_name, x=x, y=y, z=z)
    mujoco.mj_forward(model, data)
    return (x, y, z)
