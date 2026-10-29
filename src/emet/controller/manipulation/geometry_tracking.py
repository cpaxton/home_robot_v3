# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Bounded multi-start geometry registration, with explicit pose ambiguity.

Fit residuals and numerical observability are diagnostics, not calibrated pose
confidence. A candidate is not automatically a native executor's measured pose.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from emet.controller.manipulation.geometry_models import immutable, rotations_equivalent
from emet.controller.manipulation.native_inputs import ObjectPoseEvidence, ObservationUnavailable


@dataclass(frozen=True)
class GeometryFit:
    model: object
    object_ref: str
    poses: np.ndarray
    rms_m: np.ndarray
    nullspaces: tuple[np.ndarray, ...]
    observation_ids: tuple[str, ...]
    received_monotonic: float
    evaluations: int
    search_boundary_hit: bool
    search_starts: int = 0

    def __post_init__(self):
        for name in ("poses", "rms_m"):
            object.__setattr__(self, name, immutable(getattr(self, name)))
        object.__setattr__(self, "nullspaces", tuple(immutable(v) for v in self.nullspaces))

    @property
    def translation_observable(self):
        """Local rank diagnostic only; distinct retained centers remain possible."""
        return all(not len(space) or np.max(np.linalg.norm(space[:, :3], axis=1)) < 0.05 for space in self.nullspaces)

    @property
    def orientation_observable(self):
        return self.model.symmetry == "undeclared" and len(self.poses) == 1 and all(not len(v) for v in self.nullspaces)

    def require_unique(self, *, now=None, position_error_m=None, rotation_error_rad=None):
        now = time.monotonic() if now is None else now
        if (
            position_error_m is None
            or rotation_error_rad is None
            or not np.isfinite([position_error_m, rotation_error_rad]).all()
            or not 0 < position_error_m <= 0.01
            or not 0 < rotation_error_rad <= 0.05
        ):
            raise ObservationUnavailable("geometry_tracking_uncalibrated")
        if not 0 <= now - self.received_monotonic <= 2:
            raise ObservationUnavailable("object_observation_stale")
        if (
            self.search_boundary_hit
            or not self.translation_observable
            or not self.orientation_observable
            or len(self.poses) != 1
        ):
            raise ObservationUnavailable("object_pose_ambiguous")
        return ObjectPoseEvidence(self.poses[0], self.received_monotonic, "rgbd_tracking", self.observation_ids[-1])

    def candidate_bounds(self):
        """Union of retained models; sampled candidates are NOT an uncertainty certificate."""
        boxes = np.array([self.model.world_bounds(p[:3, :3], p[:3, 3]) for p in self.poses])
        return np.array([boxes[:, 0].min(0), boxes[:, 1].max(0)])

    def as_dict(self):
        return {
            "state": "candidate",
            "construction": self.model.construction,
            "model_id": self.model.model_id,
            "observation_ids": list(self.observation_ids),
            "received_monotonic": self.received_monotonic,
            "symmetry": self.model.symmetry,
            "hypotheses": len(self.poses),
            "translation_observable": bool(self.translation_observable),
            "orientation_observable": bool(self.orientation_observable),
            "rms_m": self.rms_m.tolist(),
            "centers": [p[:3, 3].tolist() for p in self.poses],
            "poses": self.poses.tolist(),
            "nullspaces": [space.tolist() for space in self.nullspaces],
            "center_spread_m": np.ptp(self.poses[:, :3, 3], axis=0).tolist(),
            "candidate_bounds": self.candidate_bounds().tolist(),
            "search_boundary_hit": self.search_boundary_hit,
            "evaluations": self.evaluations,
            "search_starts": self.search_starts,
            "execution_qualified": False,
            "observability_scope": "numerical_local_diagnostic_not_calibrated_confidence",
        }


def fit_geometry(
    points_world,
    model,
    *,
    object_ref,
    observation_ids,
    received_monotonic,
    max_residual_m=0.003,
    max_points=800,
    max_starts=24,
    max_evaluations=6000,
    initial_poses=(),
):
    """Fit declared geometry to freshly segmented measured 3D surfaces.

    Initial poses only seed search. They are never observations or a fallback.
    All accepted points must lie near the model; do not trim background into a
    target. Small/tiny patches and unobservable directions cannot become unique.
    """
    points = np.asarray(points_world, float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 25
        or not np.isfinite(points).all()
        or not object_ref
        or not observation_ids
        or not np.isfinite(received_monotonic)
        or not 0 < max_residual_m <= 0.01
        or type(max_starts) is not int
        or not 1 <= max_starts <= 64
        or type(max_evaluations) is not int
        or max_evaluations < 20
        or max_points < 25
    ):
        raise ObservationUnavailable("geometry_tracking_inputs_unavailable")
    if np.linalg.svd(points - points.mean(0), compute_uv=False)[1] / np.sqrt(len(points)) < 0.004:
        raise ObservationUnavailable("geometry_tracking_unobservable")
    # Uniform deterministic sampling controls work; all original points are
    # subsequently validated, including any skipped foreground/background outlier.
    sample = points[np.linspace(0, len(points) - 1, min(len(points), max_points)).astype(int)]
    center = (points.min(0) + points.max(0)) / 2
    radius = float(np.linalg.norm(model.half_extents))
    low, high = center - 2 * radius, center + 2 * radius
    seeds = []
    for pose in initial_poses:
        pose = np.asarray(pose, float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("invalid_geometry_search_seed")
        seeds.append(np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_rotvec()])
    rotations = Rotation.create_group("O").as_rotvec()
    for vector in rotations:
        seeds.append(np.r_[center, vector])
    unique = []
    for seed in seeds:
        rotation = Rotation.from_rotvec(seed[3:]).as_matrix()
        if any(
            np.allclose(old[:3], seed[:3], atol=1e-10, rtol=0)
            and rotations_equivalent(model, Rotation.from_rotvec(old[3:]).as_matrix(), rotation)
            for old in unique
        ):
            continue
        unique.append(seed)
    seeds = unique[:max_starts]
    candidates = []
    evaluations = 0
    boundary = False
    scale = max(radius, 0.001)

    def residual(parameters, data=sample):
        nonlocal evaluations
        evaluations += 1
        if evaluations > max_evaluations:
            raise ObservationUnavailable("geometry_tracking_budget_exhausted")
        rotation = Rotation.from_rotvec(parameters[3:]).as_matrix()
        return model.distance((data - parameters[:3]) @ rotation)

    for seed in seeds:
        seed[:3] = np.clip(seed[:3], low + 1e-7, high - 1e-7)
        result = least_squares(
            residual,
            seed,
            bounds=(np.r_[low, [-np.inf] * 3], np.r_[high, [np.inf] * 3]),
            loss="soft_l1",
            f_scale=max_residual_m,
            x_scale=[scale] * 3 + [1.0] * 3,
            max_nfev=100,
        )
        if not result.success:
            continue
        rotation = Rotation.from_rotvec(result.x[3:]).as_matrix()
        errors = model.validation_distance((points - result.x[:3]) @ rotation)
        if np.quantile(errors, 0.95) > max_residual_m or errors.max() > 3 * max_residual_m:
            continue
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = result.x[:3]
        # Perturb about the fitted MODEL origin, using metres/radians scaled by
        # model radius, to expose continuous null directions without image texture.
        local = (sample - result.x[:3]) @ rotation
        columns = []
        epsilon = 1e-4
        for axis in range(6):
            delta = np.zeros(3)
            delta[axis % 3] = epsilon
            if axis < 3:
                a = (sample - (result.x[:3] + delta)) @ rotation
                b = (sample - (result.x[:3] - delta)) @ rotation
            else:
                a = local @ Rotation.from_rotvec(delta / scale).as_matrix()
                b = local @ Rotation.from_rotvec(-delta / scale).as_matrix()
            columns.append((model.distance(a) - model.distance(b)) / (2 * epsilon))
        _, singular, vt = np.linalg.svd(np.column_stack(columns), full_matrices=False)
        null = vt[singular < max(0.02 * singular[0], 0.01)]
        # Keep distinct centers/axes and all undeclared rotational alternatives.
        duplicate = False
        for old, _, _ in candidates:
            same_center = np.linalg.norm(old[:3, 3] - pose[:3, 3]) < 0.001
            if model.symmetry == "all_rotations":
                same_orientation = True
            elif model.symmetry == "axial_unoriented":
                same_orientation = abs(old[:3, 2] @ pose[:3, 2]) > np.cos(0.01)
            else:
                same_orientation = np.trace(old[:3, :3].T @ rotation) > 1 + 2 * np.cos(0.01)
            if same_center and same_orientation:
                duplicate = True
                break
        if not duplicate:
            candidates.append((pose, float(np.sqrt(np.mean(errors**2))), null))
        boundary |= bool(np.any(result.x[:3] - low < 0.001) or np.any(high - result.x[:3] < 0.001))
    if not candidates:
        raise ObservationUnavailable("geometry_tracking_no_match")
    candidates.sort(key=lambda c: c[1])
    return GeometryFit(
        model,
        object_ref,
        np.array([c[0] for c in candidates]),
        np.array([c[1] for c in candidates]),
        tuple(c[2] for c in candidates),
        tuple(observation_ids),
        received_monotonic,
        evaluations,
        boundary,
        len(seeds),
    )


def fit_geometry_views(views, model, *, scene_revision, **kwargs):
    """Aggregate measured surfaces only within one declared static object epoch."""
    views = tuple(views)
    if not views or any(v["scene_revision"] != scene_revision for v in views):
        raise ObservationUnavailable("geometry_tracking_scene_changed")
    return fit_geometry(
        np.concatenate([v["points_world"] for v in views]),
        model,
        observation_ids=tuple(v["observation_id"] for v in views),
        received_monotonic=min(v["received_monotonic"] for v in views),
        **kwargs,
    )


class GeometryTracker:
    """Fixed model, fresh independently segmented points, measured-pose search seeds."""

    input_mode = "observed"

    def __init__(self, model, *, object_ref, max_evaluations=6000):
        self.model, self.object_ref, self.max_evaluations = model, object_ref, max_evaluations
        self.latest = None
        self._received = -np.inf
        self._seen = set()
        self._epoch = None

    def invalidate(self):
        self.latest = None
        self._epoch = None

    def update(self, points_world, *, observation_id, received_monotonic, scene_epoch, now):
        previous, self.latest = self.latest, None
        if (
            not scene_epoch
            or not observation_id
            or observation_id in self._seen
            or not np.isfinite([now, received_monotonic]).all()
            or not self._received < received_monotonic <= now
            or now - received_monotonic > 2
        ):
            raise ObservationUnavailable("object_observation_stale")
        self._received = received_monotonic
        self._seen.add(observation_id)
        if len(self._seen) > 4096:
            raise ObservationUnavailable("geometry_tracking_session_exhausted")
        if self._epoch is not None and self._epoch != scene_epoch:
            raise ObservationUnavailable("geometry_tracking_scene_changed")
        self._epoch = scene_epoch
        seeds = () if previous is None else previous.poses[:12]
        self.latest = fit_geometry(
            points_world,
            self.model,
            object_ref=self.object_ref,
            observation_ids=(observation_id,),
            received_monotonic=received_monotonic,
            initial_poses=seeds,
            max_evaluations=self.max_evaluations,
        )
        return self.latest
