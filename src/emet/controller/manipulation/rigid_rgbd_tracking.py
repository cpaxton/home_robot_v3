# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Fixed-reference RGBD material-landmark tracking for opaque rigid objects.

Image features establish correspondences; a low nearest-surface ICP residual
alone cannot identify a pose. Textureless, occluded or poorly spread features
return unknown. This is a bounded local tracker, not global re-identification.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import ConvexHull

from emet.controller.manipulation.native_inputs import ObjectPoseEvidence, ObservationUnavailable
from emet.controller.manipulation.observed_collision import PayloadGeometryEvidence


def _points(value):
    points = np.asarray(value, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('invalid_rigid_landmarks')
    return points


def fit_rigid_landmarks(reference, current, *, tolerance_m=.004, min_inliers=12, max_trials=128):
    """RANSAC and rigid least squares on *paired* independently tracked landmarks.

    Reject collinear/tiny patches. Planar textured features can determine rigid
    motion; this does not assign a unique object yaw to an untextured surface.
    The returned residual is a diagnostic, not a statistical confidence bound.
    """
    reference, current = _points(reference), _points(current)
    if (reference.shape != current.shape or len(reference) < min_inliers
            or min_inliers < 6 or tolerance_m <= 0 or not np.isfinite(tolerance_m)
            or type(max_trials) is not int or not 1 <= max_trials <= 1024):
        raise ObservationUnavailable('rigid_tracking_insufficient')

    def fit(a, b):
        ca, cb = a.mean(0), b.mean(0)
        u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
        r = vt.T @ np.diag([1., 1., np.linalg.det(vt.T @ u.T)]) @ u.T
        return r, cb - r @ ca

    best = np.zeros(len(reference), dtype=bool)
    rng = np.random.default_rng(0)
    for _ in range(max_trials):
        indices = rng.choice(len(reference), 3, replace=False)
        a = reference[indices]
        if np.linalg.svd(a - a.mean(0), compute_uv=False)[1] < .001:
            continue
        r, t = fit(a, current[indices])
        mask = np.linalg.norm(reference @ r.T + t - current, axis=1) <= tolerance_m
        if mask.sum() > best.sum():
            best = mask
    if best.sum() < min_inliers or best.mean() < .8:
        raise ObservationUnavailable('rigid_tracking_inconsistent')
    for points in (reference[best], current[best]):
        singular = np.linalg.svd(points - points.mean(0), compute_uv=False) / np.sqrt(best.sum())
        if singular[1] < .006 or singular[1] / singular[0] < .1:
            raise ObservationUnavailable('rigid_tracking_unobservable')
    r, t = fit(reference[best], current[best])
    errors = np.linalg.norm(reference[best] @ r.T + t - current[best], axis=1)
    if errors.max() > tolerance_m:
        raise ObservationUnavailable('rigid_tracking_inconsistent')
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = r, t
    return pose, float(errors.max()), best


@dataclass(frozen=True)
class RigidTrack:
    payload: PayloadGeometryEvidence
    registration_residual_m: float
    inlier_count: int
    position_error_m: float
    rotation_error_rad: float


class RigidRgbdTracker:
    """RGB feature flow + measured depth, always against one immutable reference.

    A supplied complete envelope is a declared geometry prior, independent of any
    simulator body. Error bounds are externally calibrated admission limits,
    never inferred as zero from a good fit. Caller provides current object masks;
    losing segmentation does not fall back to the old mask or predicted EE pose.
    """
    input_mode = 'observed'

    def __init__(self, *, object_ref, reference_id, geometry_revision, rgb, world_xyz, mask,
                 world_from_reference, vertices_reference, received_monotonic,
                 position_error_m, rotation_error_rad, reference_max_age_s=30.):
        if (not 0 < position_error_m <= .01 or not 0 < rotation_error_rad <= .05
                or not np.isfinite([position_error_m, rotation_error_rad, received_monotonic, reference_max_age_s]).all()
                or not 0 < reference_max_age_s <= 30):
            raise ValueError('invalid_tracking_calibration')
        self._validate(rgb, world_xyz, mask)
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        interior = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8))
        pixels = cv2.goodFeaturesToTrack(self.gray, maxCorners=128, qualityLevel=.02, minDistance=4,
                                        mask=interior)
        if pixels is None or len(pixels) < 12:
            raise ObservationUnavailable('rigid_tracking_insufficient')
        pixels = pixels.astype(np.float32)
        patches = np.stack([cv2.getRectSubPix(self.gray, (9, 9), tuple(p[0])).ravel() for p in pixels]).astype(float)
        distances = np.sqrt(np.mean((patches[:, None] - patches[None, :]) ** 2, axis=-1))
        np.fill_diagonal(distances, np.inf)
        unique = distances.min(axis=1) > 8.
        self.pixels = pixels[unique]
        if len(self.pixels) < 12:
            raise ObservationUnavailable('rigid_tracking_unobservable')
        xy = np.rint(self.pixels[:, 0]).astype(int)
        world = world_xyz[xy[:, 1], xy[:, 0]]
        initial = ObjectPoseEvidence(world_from_reference, received_monotonic, 'rgbd_tracking', 'reference')
        pose = initial.world_from_object
        self.reference = (world - pose[:3, 3]) @ pose[:3, :3]
        self.vertices = PayloadGeometryEvidence(object_ref, reference_id, geometry_revision,
            vertices_reference, initial, 'known_model_prior').vertices_reference
        # The independently supplied prior must contain all measured masked points.
        measured = (world_xyz[mask] - pose[:3, 3]) @ pose[:3, :3]
        hull = ConvexHull(self.vertices)
        if np.any(measured @ hull.equations[:, :3].T + hull.equations[:, 3] > position_error_m):
            raise ObservationUnavailable('payload_prior_inconsistent')
        self.object_ref, self.reference_id, self.geometry_revision = object_ref, reference_id, geometry_revision
        self.reference_received, self.reference_max_age_s = received_monotonic, reference_max_age_s
        self.position_error_m, self.rotation_error_rad = position_error_m, rotation_error_rad
        self._latest_received, self._latest_capture, self.latest = received_monotonic, None, None

    @staticmethod
    def _validate(rgb, world_xyz, mask):
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
                or np.shape(world_xyz) != rgb.shape or np.shape(mask) != rgb.shape[:2] or mask.dtype != bool
                or not mask.any() or not np.isfinite(world_xyz[mask]).all()):
            raise ObservationUnavailable('rigid_tracking_inputs_unavailable')

    def update(self, *, rgb, world_xyz, mask, capture_id, received_monotonic, now):
        # A failed update invalidates cached output even when the old pose is fresh.
        self.latest = None
        if (not capture_id or capture_id == self._latest_capture
                or not np.isfinite([received_monotonic, now]).all()
                or not self._latest_received < received_monotonic <= now
                or now - received_monotonic > 2 or now - self.reference_received > self.reference_max_age_s):
            raise ObservationUnavailable('object_observation_stale')
        self._latest_capture, self._latest_received = capture_id, received_monotonic
        self._validate(rgb, world_xyz, mask)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        forward, ok, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.pixels, None)
        if forward is None:
            raise ObservationUnavailable('rigid_tracking_insufficient')
        backward, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, forward, None)
        if backward is None:
            raise ObservationUnavailable('rigid_tracking_insufficient')
        valid = ok.ravel().astype(bool) & back_ok.ravel().astype(bool)
        valid &= np.linalg.norm(backward[:, 0] - self.pixels[:, 0], axis=1) <= .5
        xy = np.rint(forward[:, 0]).astype(int)
        valid &= np.all(xy >= 0, axis=1) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] < mask.shape[0])
        selected = np.flatnonzero(valid)
        interior = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        selected = selected[interior[xy[selected, 1], xy[selected, 0]]]
        current = world_xyz[xy[selected, 1], xy[selected, 0]]
        pose, residual, inliers = fit_rigid_landmarks(self.reference[selected], current,
                                                   tolerance_m=min(.004, self.position_error_m))
        # Photometric consistency of paired landmarks adds an independent check
        # against LK locking onto a nearby, differently textured surface.
        ref_xy = np.rint(self.pixels[selected, 0]).astype(int)
        difference = np.abs(gray[xy[selected, 1], xy[selected, 0]].astype(float)
                            - self.gray[ref_xy[:, 1], ref_xy[:, 0]].astype(float))
        if np.median(difference[inliers]) > 25:
            raise ObservationUnavailable('rigid_tracking_inconsistent')
        payload = PayloadGeometryEvidence(self.object_ref, self.reference_id, self.geometry_revision,
            self.vertices, ObjectPoseEvidence(pose, received_monotonic, 'rgbd_tracking', capture_id), 'known_model_prior')
        self.latest = RigidTrack(payload, residual, int(inliers.sum()), self.position_error_m, self.rotation_error_rad)
        return self.latest

    def observe_payload(self, object_ref):
        return self.latest.payload if self.latest is not None and object_ref == self.object_ref else None
