# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Conservative scalar-joint FK intervals, without sampling unmeasured states."""

import mujoco
import numpy as np

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.motion.placement_geometry import box_corners


def _product(a, b):
    al, ah = a
    bl, bh = b
    terms = np.stack(
        (
            al[:, :, None] * bl[None, :, :],
            al[:, :, None] * bh[None, :, :],
            ah[:, :, None] * bl[None, :, :],
            ah[:, :, None] * bh[None, :, :],
        )
    )
    return terms.min(0).sum(1), terms.max(0).sum(1)


def _vector(rotation, vector):
    lo, hi = _product(rotation, (vector[0][:, None], vector[1][:, None]))
    return lo[:, 0], hi[:, 0]


def _trig(lo, hi, *, cosine=False):
    if not np.isfinite([lo, hi]).all() or hi - lo >= 2 * np.pi:
        return -1.0, 1.0
    if lo > hi:
        raise ValueError("invalid_joint_interval")
    offset = 0.0 if cosine else np.pi / 2
    values = [np.cos(lo), np.cos(hi)] if cosine else [np.sin(lo), np.sin(hi)]
    for k in range(int(np.ceil((lo - offset) / np.pi)), int(np.floor((hi - offset) / np.pi)) + 1):
        values.append((-1.0) ** k)
    return min(values), max(values)


def _hinge(axis, lo, hi):
    fixed = np.outer(axis, axis)
    cosine = np.eye(3) - fixed
    x, y, z = axis
    sine = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    cl, ch = _trig(lo, hi, cosine=True)
    sl, sh = _trig(lo, hi)
    lower = fixed + np.minimum(cosine * cl, cosine * ch) + np.minimum(sine * sl, sine * sh)
    upper = fixed + np.maximum(cosine * cl, cosine * ch) + np.maximum(sine * sl, sine * sh)
    return np.clip(lower, -1, 1), np.clip(upper, -1, 1)


def _fixed_quaternion(quat):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat)
    matrix = matrix.reshape(3, 3)
    return matrix, matrix


def subtree_geometry_intervals(model, *, root_id, world_from_root, joint_intervals, body_overrides=None):
    """Enclose collidable geom boxes for every configuration in supplied intervals.

    Hinge extrema are analytic; unlimited hinges cover a full turn. Unbounded
    prismatic/free/ball descendants reject instead of inventing a finite volume.
    Intervals are model assumptions, not measured passive-joint configurations.
    Optional world-frame body overrides must be independently bounded by caller.
    """
    body_overrides = {} if body_overrides is None else body_overrides
    rotation, translation = world_from_root[:3, :3], world_from_root[:3, 3]
    poses = {root_id: ((rotation, rotation), (translation, translation))}
    for bid in range(root_id + 1, model.nbody):
        parent = int(model.body_parentid[bid])
        if parent not in poses:
            continue
        if bid in body_overrides:
            poses[bid] = body_overrides[bid]
            continue
        rp, tp = poses[parent]
        body_offset = _vector(rp, (model.body_pos[bid], model.body_pos[bid]))
        t = tp[0] + body_offset[0], tp[1] + body_offset[1]
        r = _product(rp, _fixed_quaternion(model.body_quat[bid]))
        r = np.clip(r[0], -1, 1), np.clip(r[1], -1, 1)
        for jid in range(int(model.body_jntadr[bid]), int(model.body_jntadr[bid] + model.body_jntnum[bid])):
            if jid not in joint_intervals:
                raise ObservationUnavailable("robot_joint_interval_unavailable")
            low, high = joint_intervals[jid]
            if not np.isfinite([low, high]).all() or low > high:
                raise ValueError("invalid_joint_interval")
            address = int(model.jnt_qposadr[jid])
            low, high = low - model.qpos0[address], high - model.qpos0[address]
            kind = model.jnt_type[jid]
            if kind == mujoco.mjtJoint.mjJNT_SLIDE:
                delta = (
                    np.minimum(model.jnt_axis[jid] * low, model.jnt_axis[jid] * high),
                    np.maximum(model.jnt_axis[jid] * low, model.jnt_axis[jid] * high),
                )
                offset = _vector(r, delta)
                t = t[0] + offset[0], t[1] + offset[1]
            elif kind == mujoco.mjtJoint.mjJNT_HINGE:
                pivot = model.jnt_pos[jid]
                old_pivot = _vector(r, (pivot, pivot))
                new_r = _product(r, _hinge(model.jnt_axis[jid], low, high))
                r = np.clip(new_r[0], -1, 1), np.clip(new_r[1], -1, 1)
                new_pivot = _vector(r, (pivot, pivot))
                t = t[0] + old_pivot[0] - new_pivot[1], t[1] + old_pivot[1] - new_pivot[0]
            else:
                raise ObservationUnavailable("robot_joint_interval_type_unsupported")
        poses[bid] = r, t
    boxes, geom_ids = [], []
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        if bid not in poses or not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        center, half = model.geom_aabb[gid, :3], model.geom_aabb[gid, 3:]
        corners = box_corners([center - half, center + half])
        local_rotation = _fixed_quaternion(model.geom_quat[gid])[0]
        corners = corners @ local_rotation.T + model.geom_pos[gid]
        r, t = poses[bid]
        transformed = [_vector(r, (point, point)) for point in corners]
        lo = np.min([a for a, b in transformed], axis=0) + t[0]
        hi = np.max([b for a, b in transformed], axis=0) + t[1]
        if not np.isfinite([lo, hi]).all():
            raise ObservationUnavailable("robot_geometry_unbounded")
        boxes.append([lo - 1e-9, hi + 1e-9])
        geom_ids.append(gid)
    if not boxes:
        raise ObservationUnavailable("gripper_geometry_unknown")
    return tuple(geom_ids), np.array(boxes)
