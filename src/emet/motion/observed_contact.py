# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Bounded target contact; never a free-space or environment exemption.

The target is a complete observed outer envelope, including pose uncertainty.
Fingertip IDs name exact collision geoms, not body subtrees. An allowed contact
must remain outside the target's eroded core. Unresolved geometry rejects.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import OptimizeWarning, linprog

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.motion.observed_solids import WorldSolid

if TYPE_CHECKING:
    from emet.controller.manipulation.observed_effects import CaptureInterval


def _convex_separation(a, b, bounds):
    """Decide halfspace-envelope intersection without spatial subdivision."""
    rows, limits = [], []
    for solid in (a, b):
        if solid.kind == 'convex':
            planes = solid._planes
        else:
            normals = np.vstack((np.eye(3), -np.eye(3)))
            planes = np.column_stack((normals, -np.tile(solid.dimensions, 2)))
        normals = planes[:, :3] @ solid.world_from_local[:3, :3].T
        rows.append(normals)
        limits.append(solid.offset_m-planes[:, 3]+normals @ solid.world_from_local[:3, 3])
    # Inflate by a tiny numeric allowance before accepting infeasibility. An
    # unresolved solver outcome falls back to the existing conservative check.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected.*', category=OptimizeWarning)
        result = linprog(np.zeros(3), A_ub=np.vstack(rows), b_ub=np.concatenate(limits)+1e-8,
                         bounds=list(zip(bounds[0], bounds[1], strict=True)), method='highs-ds',
                         options={'threads': 1, 'primal_feasibility_tolerance': 1e-9,
                                  'dual_feasibility_tolerance': 1e-9})
    if result.status == 2:
        return True
    if result.success and all(solid.lower_distance(result.x[None])[0] <= 1e-8+1e-12 for solid in (a, b)):
        return False
    return None


def solids_disjoint(a: WorldSolid, b: WorldSolid, *, max_nodes=4096, min_extent_m=0.0001):
    """Prove separation with convex halfspaces or bounded subdivision.

    Both distance functions are conservative, 1-Lipschitz lower bounds. A box
    discarded by either solid cannot contain their intersection. This works for
    rotated primitives and declared convex outer envelopes without claiming that
    overlapping AABBs establish contact or certify penetration depth.
    """
    if type(max_nodes) is not int or max_nodes < 1 or not np.isfinite(min_extent_m) or min_extent_m <= 0:
        raise ValueError("invalid_contact_query_budget")
    bounds = np.stack((np.maximum(a.bounds[0], b.bounds[0]), np.minimum(a.bounds[1], b.bounds[1])))
    if np.any(bounds[0] > bounds[1]):
        return True
    if a.kind in {'convex', 'box'} and b.kind in {'convex', 'box'} and 'convex' in {a.kind, b.kind}:
        exact = _convex_separation(a, b, bounds)
        if exact is not None:
            return exact
    pending, nodes = [bounds], 0
    while pending:
        cell = pending.pop()
        nodes += 1
        if nodes > max_nodes:
            return False
        if a.disjoint_box(cell) or b.disjoint_box(cell):
            continue
        center = cell.mean(0)[None]
        if a.lower_distance(center)[0] <= 0 and b.lower_distance(center)[0] <= 0:
            return False
        axis = int(np.argmax(cell[1] - cell[0]))
        if cell[1, axis] - cell[0, axis] <= min_extent_m:
            return False
        left, right = cell.copy(), cell.copy()
        left[1, axis] = right[0, axis] = float(cell[:, axis].mean())
        pending.extend((left, right))
    return True


CONTACT_PHASES = frozenset(("grasp", "close", "lift", "preplace", "place", "release"))
MOTION_PHASES = CONTACT_PHASES | frozenset(("preflight", "pregrasp", "retreat", "gripper_open", "probe"))


@dataclass(frozen=True)
class ObservedTargetContact:
    object_ref: str
    reference_id: str
    geometry_revision: str
    capture: CaptureInterval
    target: WorldSolid
    model_signature: str
    calibration_id: str
    fingertip_geom_ids: frozenset[int]
    max_penetration_m: float
    approximate_volume: bool = False

    def __post_init__(self):
        # Deferred because effects imports the observed collision checker.
        from emet.controller.manipulation.observed_effects import CaptureInterval

        fingers = frozenset(self.fingertip_geom_ids)
        if (
            not all(
                isinstance(s, str) and s
                for s in (
                    self.object_ref,
                    self.reference_id,
                    self.geometry_revision,
                    self.model_signature,
                    self.calibration_id,
                )
            )
            or not isinstance(self.capture, CaptureInterval)
            or not isinstance(self.target, WorldSolid)
            or self.target.offset_m < 0
            or not fingers
            or any(type(g) is not int or g < 0 for g in fingers)
            or type(self.approximate_volume) is not bool
            or not np.isfinite(self.max_penetration_m)
            or not 0 <= self.max_penetration_m <= 0.005
        ):
            raise ValueError("invalid_observed_target_contact")
        object.__setattr__(self, "fingertip_geom_ids", fingers)

    @property
    def profile_identity(self):
        return (
            self.object_ref,
            self.reference_id,
            self.geometry_revision,
            self.model_signature,
            self.calibration_id,
            self.fingertip_geom_ids,
            self.max_penetration_m,
            self.capture.calibration_id,
            self.approximate_volume,
        )

    @classmethod
    def from_frame(cls, frame, *, model_signature, calibration_id, fingertip_geom_ids, max_penetration_m):
        from emet.controller.manipulation.observed_effects import pose_uncertainty_margin

        payload = frame.payload
        radius = float(np.linalg.norm(payload.vertices_reference, axis=1).max())
        margin = pose_uncertainty_margin(radius, frame.position_error_m, frame.rotation_error_rad)
        target = WorldSolid(
            "convex", (), payload.pose.world_from_object, margin, payload.vertices_reference, "observed-target-envelope"
        )
        return cls(
            payload.object_ref,
            payload.reference_id,
            payload.revision,
            frame.capture,
            target,
            model_signature,
            calibration_id,
            frozenset(fingertip_geom_ids),
            max_penetration_m,
            approximate_volume=payload.construction == 'approximate_observed_box',
        )

    def require(self, *, now, model_signature, capture_id=None):
        self.capture.require(now=now)
        if model_signature != self.model_signature:
            raise ObservationUnavailable("target_contact_identity_changed")
        if capture_id is not None and capture_id != self.capture.observation_id:
            raise ObservationUnavailable("capture_identity_mismatch")

    def collides(self, robot: WorldSolid, *, geom_id, phase, target=None):
        if phase not in MOTION_PHASES:
            raise ValueError("invalid_contact_phase")
        target = self.target if target is None else target
        if geom_id in self.fingertip_geom_ids and phase in CONTACT_PHASES:
            target = target.with_offset(target.offset_m - self.max_penetration_m)
            # A vanishing core must not turn the whole target into a contact
            # exemption. Prove a nonempty interior at a convex combination of
            # model vertices (or the primitive center), otherwise reject.
            center = (
                target.world_from_local[:3, :3] @ target.vertices.mean(0) + target.world_from_local[:3, 3]
                if target.kind == "convex"
                else target.world_from_local[:3, 3]
            )
            if target.lower_distance(center[None])[0] >= 0:
                return True
        return not solids_disjoint(robot, target)
