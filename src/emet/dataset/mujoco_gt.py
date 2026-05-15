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

"""Extract object ground truth from MuJoCo ``mjModel`` / ``mjData`` (body poses)."""

from __future__ import annotations

import fnmatch
from typing import Any

import mujoco
import numpy as np

from emet.dataset.schema import ObjectRecord


def gt_objects_for_zmq_message(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    *,
    environment: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Object GT list for ZMQ payloads; *environment* ``kind`` selects body name glob heuristics."""
    kind = (environment or {}).get("kind")
    if kind == "molmospaces":
        name_globs: tuple[str, ...] = ("object*", "*OBJ*", "*obj_*", "*Prop*")
    else:
        name_globs = ("object*",)
    return extract_gt_object_dicts(mj_model, mj_data, name_globs=name_globs)


def extract_gt_object_dicts(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    *,
    allowlist: list[str] | None = None,
    name_globs: tuple[str, ...] = ("object*",),
    skip_substring: tuple[str, ...] = ("robot", "base_link", "world"),
) -> list[dict[str, Any]]:
    """Return JSON-serializable object dicts for bodies matching *allowlist* or *name_globs*.

    World body (id 0) is never included. Names containing any *skip_substring* are skipped unless
    explicitly listed in *allowlist*.
    """
    out: list[dict[str, Any]] = []
    for bid in range(1, mj_model.nbody):
        bname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not bname:
            continue
        if allowlist is not None:
            if bname not in allowlist:
                continue
        else:
            ln = bname.lower()
            if any(s in ln for s in skip_substring):
                continue
            if not any(fnmatch.fnmatchcase(bname, pat) for pat in name_globs):
                continue

        xpos = np.asarray(mj_data.body(bname).xpos, dtype=np.float64).reshape(3)
        xquat = np.asarray(mj_data.body(bname).xquat, dtype=np.float64).reshape(4)
        aabb_min, aabb_max = _geom_world_aabb_for_body(mj_model, mj_data, bid)

        rec = ObjectRecord(
            name=bname,
            body_name=bname,
            pos_xyz=(float(xpos[0]), float(xpos[1]), float(xpos[2])),
            quat_wxyz=(float(xquat[0]), float(xquat[1]), float(xquat[2]), float(xquat[3])),
            aabb_min_xyz=aabb_min,
            aabb_max_xyz=aabb_max,
        )
        d = rec.to_json_dict()
        d.pop("schema_version", None)
        out.append(d)
    return out


def _geom_world_aabb_for_body(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    body_id: int,
) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
    """Axis-aligned world bounds from geoms attached to *body_id* (approximate)."""
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for gid in range(mj_model.ngeom):
        if int(mj_model.geom_bodyid[gid]) != body_id:
            continue
        center = np.asarray(mj_data.geom_xpos[gid], dtype=np.float64).reshape(3)
        mat = np.asarray(mj_data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
        size = np.asarray(mj_model.geom_size[gid], dtype=np.float64).reshape(3)
        corners = _obb_corners(center, mat, size)
        mins.append(corners.min(axis=0))
        maxs.append(corners.max(axis=0))
    if not mins:
        return None, None
    lo = np.min(np.stack(mins, axis=0), axis=0)
    hi = np.max(np.stack(maxs, axis=0), axis=0)
    return (
        (float(lo[0]), float(lo[1]), float(lo[2])),
        (float(hi[0]), float(hi[1]), float(hi[2])),
    )


def _obb_corners(center: np.ndarray, rot: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Eight corners of an axis-aligned box in geom frame, rotated to world."""
    hx, hy, hz = float(half[0]), float(half[1]), float(half[2])
    local = np.array(
        [
            [-hx, -hy, -hz],
            [-hx, -hy, hz],
            [-hx, hy, -hz],
            [-hx, hy, hz],
            [hx, -hy, -hz],
            [hx, -hy, hz],
            [hx, hy, -hz],
            [hx, hy, hz],
        ],
        dtype=np.float64,
    )
    return center.reshape(1, 3) + (rot @ local.T).T
