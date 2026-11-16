# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Rank existing pinch poses by opposing box-face alignment, without admission.

This is a geometric preference, not an antipodal/contact or force certificate.
Unknown/compound geometry retains the caller's order. Every returned candidate
still needs the ordinary IK, collision, execution and retention checks.
"""
from __future__ import annotations

import mujoco
import numpy as np

from emet.motion.mujoco_collision import body_subtree


def face_aligned_grasp_order(model, data, object_body, *, closing_axis, candidates, best_only=False):
    """Prefer jaw directions normal to a single collision box's world faces.

    Keep the exact candidate population and stable prior order within equal
    alignment. Use geom orientation, including welded children/local rotation;
    no category labels, body-name special cases, state writes or physics steps.
    """
    candidates = list(candidates)
    bodies = body_subtree(model, object_body)
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies
             and (model.geom_contype[g] or model.geom_conaffinity[g])]
    if len(geoms) != 1 or model.geom_type[geoms[0]] != mujoco.mjtGeom.mjGEOM_BOX:
        return candidates
    axis = np.asarray(closing_axis, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
        raise ValueError('invalid_gripper_closing_axis')
    axis = axis / np.linalg.norm(axis)
    normals = data.geom_xmat[geoms[0]].reshape(3, 3)
    # Preserve the prior pitch/height preference among nearly aligned poses.
    # Numerical/settling differences near cosine 1 must not promote a fallback
    # frame over a preferred pinch. This is ranking resolution, not admission.
    def cost(candidate):
        return -round(float(np.max(np.abs(normals.T @ candidate[0] @ axis))), 3)
    ordered = sorted(candidates, key=cost)
    if best_only and ordered:
        return [candidate for candidate in ordered if cost(candidate) == cost(ordered[0])]
    return ordered
