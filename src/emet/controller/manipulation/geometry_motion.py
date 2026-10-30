# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Conservative geometry co-motion diagnostics; temporal effects remain separate."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from emet.controller.manipulation.observed_effects import rigid_copy, rotation_distance


def check_geometry_comotion(
    reference,
    current,
    *,
    ee_before,
    ee_after,
    position_error_m,
    rotation_error_rad,
    require_lift=False,
    support_top_z=None,
):
    """Check ALL retained pose-pair hypotheses, quotienting declared shape symmetry.

    No capture/encoder alignment or contact is inferred. 'consistent' is geometric
    evidence only, never a physical-tool success receipt. Inputs are measured
    fits; commanded transforms may not substitute for either fit or measured EE.
    """
    result = {"state": "unknown", "reason": "geometry_ambiguous", "execution_qualified": False}
    if (
        reference.object_ref != current.object_ref
        or reference.model.model_id != current.model.model_id
        or reference.received_monotonic >= current.received_monotonic
        or set(reference.observation_ids).intersection(current.observation_ids)
    ):
        return {**result, "reason": "geometry_identity_or_order_mismatch"}
    if (
        not np.isfinite([position_error_m, rotation_error_rad]).all()
        or not 0 < position_error_m <= 0.01
        or not 0 < rotation_error_rad <= 0.05
    ):
        return {**result, "reason": "geometry_error_bounds_unavailable"}
    if (
        not reference.translation_observable
        or not current.translation_observable
        or reference.search_boundary_hit
        or current.search_boundary_hit
    ):
        return result
    model = reference.model
    for fit in (reference, current):
        for space in fit.nullspaces:
            for vector in space:
                if model.symmetry == "all_rotations":
                    continue
                if model.symmetry == "axial_unoriented" and np.linalg.norm(vector[3:5]) < 0.05:
                    continue
                return result
    before, after = np.linalg.inv(rigid_copy(ee_before)), np.linalg.inv(rigid_copy(ee_after))
    if model.symmetry == "finite_box":
        group = [
            r
            for r in Rotation.create_group("O").as_matrix()
            if np.allclose(np.abs(r) @ model.half_extents, model.half_extents)
        ]
    else:
        group = [np.eye(3)]
    matches = []
    distances = []
    for p in reference.poses:
        for q in current.poses:
            a, b = before @ p, after @ q
            distance = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
            if model.symmetry == "all_rotations":
                angle = 0.0
            elif model.symmetry == "axial_unoriented":
                angle = float(np.arccos(np.clip(abs(a[:3, 2] @ b[:3, 2]), 0, 1)))
            else:
                angle = min(rotation_distance(a[:3, :3] @ r, b[:3, :3]) for r in group)
            distances.append(distance)
            matches.append(distance + 2 * position_error_m <= 0.02 and angle + 2 * rotation_error_rad <= 0.1)
    if not all(matches):
        return {
            **result,
            "state": "inconsistent" if min(distances) - 2 * position_error_m > 0.02 else "unknown",
            "translation_residual_max_m": max(distances),
        }
    if require_lift:
        if support_top_z is None or not np.isfinite(support_top_z):
            return result
        minimum_lift = float(current.poses[:, :3, 3][:, 2].min() - reference.poses[:, :3, 3][:, 2].max())
        bottom = float(current.candidate_bounds()[0, 2]) - position_error_m
        if (
            minimum_lift - 2 * position_error_m < 0.04
            or rigid_copy(ee_after)[2, 3] - rigid_copy(ee_before)[2, 3] < 0.04
            or bottom <= support_top_z + 0.02
        ):
            return {**result, "state": "inconsistent", "reason": "geometry_lift_not_supported"}
    return {
        "state": "consistent",
        "reason": "geometry_comotion_consistent",
        "execution_qualified": False,
        "translation_residual_max_m": max(distances),
        "symmetry": model.symmetry,
    }
