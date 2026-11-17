# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Finite payload/support contact backed by original depth and explicit priors.

A support label never makes a voxel free. The witness can refine only that
support's exclusive occupied cells; unrelated and unknown cells retain their
ordinary checks. Hidden backside material requires a declared solid top layer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.motion.observed_support import ObservedSupportPatch
from emet.motion.placement_geometry import box_corners

if TYPE_CHECKING:
    from emet.controller.manipulation.observed_effects import CaptureInterval


@dataclass(frozen=True)
class ObservedSupportContact:
    object_ref: str
    support_ref: str
    scene_epoch: str
    calibration_id: str
    patch: ObservedSupportPatch
    capture: CaptureInterval
    depth: np.ndarray
    intrinsic: np.ndarray
    world_from_camera: np.ndarray
    support_mask: np.ndarray
    payload_motion_error_m: float
    backing_depth_m: float = 0.0
    backing_prior_id: str = ""
    pixel_error_px: float = 0.0
    max_penetration_m: float = 0.001
    _digest: str = field(init=False, repr=False)

    def __post_init__(self):
        from emet.controller.manipulation.observed_effects import CaptureInterval, rigid_copy

        depth, k, mask = np.asarray(self.depth, float), np.asarray(self.intrinsic, float), np.asarray(self.support_mask)
        if (
            not all(
                isinstance(x, str) and x
                for x in (self.object_ref, self.support_ref, self.scene_epoch, self.calibration_id)
            )
            or self.object_ref == self.support_ref
            or not isinstance(self.patch, ObservedSupportPatch)
            or not isinstance(self.capture, CaptureInterval)
            or self.capture.observation_id != self.patch.observation_id
            or self.capture.received_monotonic != self.patch.received_monotonic
            or depth.ndim != 2
            or mask.shape != depth.shape
            or mask.dtype != bool
            or not mask.any()
            or k.shape != (3, 3)
            or not np.isfinite(k).all()
            or min(k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[2], [0, 0, 1])
            or not np.allclose([k[0, 1], k[1, 0]], 0)
            or not np.isfinite(depth[mask]).all()
            or np.any(depth[mask] <= 0.1)
            or not np.isfinite(
                [self.backing_depth_m, self.pixel_error_px, self.max_penetration_m, self.payload_motion_error_m]
            ).all()
            or self.backing_depth_m < 0
            or self.pixel_error_px < 0
            or self.payload_motion_error_m < 0
            or not 0 <= self.max_penetration_m <= 0.001
            or not isinstance(self.backing_prior_id, str)
            or bool(self.backing_depth_m) != bool(self.backing_prior_id)
        ):
            raise ValueError("invalid_observed_support_contact")
        pose = rigid_copy(self.world_from_camera)
        y, x = np.where(mask)
        rays = np.stack((x, y, np.ones(len(x))), axis=-1) @ np.linalg.inv(k).T
        points = (rays * depth[y, x, None]) @ pose[:3, :3].T + pose[:3, 3]
        normal, offset = self.unit_plane
        if (
            np.max(np.abs(points @ normal - offset)) > self.patch.measurement_margin_m
            or pose[:3, 3] @ normal - offset <= self.patch.measurement_margin_m
        ):
            raise ValueError("invalid_observed_support_witness")
        digest = hashlib.sha256()
        for name, value in (("depth", depth), ("intrinsic", k), ("world_from_camera", pose), ("support_mask", mask)):
            frozen = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
            object.__setattr__(self, name, frozen)
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        digest.update(self.patch.bounds.tobytes() + self.patch.plane.tobytes())
        object.__setattr__(self, "_digest", digest.hexdigest())

    @property
    def unit_plane(self):
        a, b, c = self.patch.plane
        norm = float(np.linalg.norm([-a, -b, 1.0]))
        return np.array([-a, -b, 1.0]) / norm, c / norm

    @property
    def profile_identity(self):
        return (
            self.object_ref,
            self.support_ref,
            self.scene_epoch,
            self.calibration_id,
            self.capture,
            self._digest,
            self.patch.measurement_margin_m,
            self.backing_depth_m,
            self.backing_prior_id,
            self.pixel_error_px,
            self.max_penetration_m,
            self.payload_motion_error_m,
        )

    def require(self, background, *, now, static_assumption=None):
        if (
            background.scene_epoch != self.scene_epoch
            or background.latest_capture.calibration_id != self.capture.calibration_id
            or self.capture.observation_id not in background.capture_ids
            or self.capture not in background.capture_history
            or self.capture.latest > background.latest_capture.latest
        ):
            raise ObservationUnavailable("support_geometry_unknown")
        if static_assumption is None:
            self.capture.require(now=now)
        elif (
            static_assumption.scene_epoch != self.scene_epoch
            or not static_assumption.started_monotonic
            <= self.capture.earliest
            <= now
            <= static_assumption.expires_monotonic
            or self.capture.received_monotonic > now
        ):
            raise ObservationUnavailable("support_geometry_unknown")

    def physical_clearance(self, physical):
        """Worst-case signed distance, retaining the payload and plane errors."""
        normal, offset = self.unit_plane
        return float(np.min(box_corners(physical.bounds) @ normal - offset) - self.patch.measurement_margin_m)

    def eligible(self, physical, *, object_ref, phase, departure_clearance_m=None):
        if object_ref != self.object_ref or phase not in ("lift", "place", "release"):
            return False
        margin = self.patch.measurement_margin_m
        bounds = physical.bounds
        if np.any(bounds[0, :2] < self.patch.bounds[0, :2] + margin) or np.any(
            bounds[1, :2] > self.patch.bounds[1, :2] - margin
        ):
            return False
        floor = -self.max_penetration_m
        if phase == "lift":
            if departure_clearance_m is None or not np.isfinite(departure_clearance_m):
                return False
            floor = max(floor, departure_clearance_m)
        return self.physical_clearance(physical) >= floor - 1e-12

    def covers_box(self, bounds, *, allow_support_material=True):
        """Prove free-or-permitted-support for a subbox at the measured support.

        Before the nearest possible return is measured free. A box reaching
        behind the return additionally needs the named finite solid-layer prior.
        This coupled ray/material proof never changes an occupancy cell to free.
        Calibration budgets must cover axial depth, world plane/footprint and
        pixel reprojection errors; matching pixels alone do not qualify them.
        """
        margin = self.patch.measurement_margin_m
        if np.any(bounds[0, :2] < self.patch.bounds[0, :2] + margin) or np.any(
            bounds[1, :2] > self.patch.bounds[1, :2] - margin
        ):
            return False
        corners = box_corners(bounds)
        camera = (corners - self.world_from_camera[:3, 3]) @ self.world_from_camera[:3, :3]
        if np.any(camera[:, 2] <= 0):
            return False
        uv = camera @ self.intrinsic.T
        uv = uv[:, :2] / uv[:, 2:]
        low = np.floor(uv.min(0) - self.pixel_error_px + 0.5).astype(int)
        high = np.floor(uv.max(0) + self.pixel_error_px + 0.5).astype(int)
        h, w = self.depth.shape
        if np.any(low < 0) or high[0] >= w or high[1] >= h:
            return False
        region = (slice(low[1], high[1] + 1), slice(low[0], high[0] + 1))
        if not self.support_mask[region].all():
            return False
        if camera[:, 2].max() + margin < self.depth[region].min():
            return True
        if not allow_support_material or not self.backing_prior_id:
            return False
        normal, offset = self.unit_plane
        return bool(np.min(corners @ normal - offset) >= margin - self.backing_depth_m)
