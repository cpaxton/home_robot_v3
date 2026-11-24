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
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.motion.guarded_placement_probe import MAX_TRAVEL_M
from emet.motion.observed_support import ObservedSupportPatch
from emet.motion.placement_geometry import bounds_array, box_corners

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

    @cached_property
    def _measured_surface_tree(self):
        y, x = np.nonzero(self.support_mask)
        rays = np.column_stack((x, y, np.ones(len(x)))) @ np.linalg.inv(self.intrinsic).T
        points = (rays*self.depth[y, x, None]) @ self.world_from_camera[:3, :3].T + self.world_from_camera[:3, 3]
        return cKDTree(points)

    @cached_property
    def measured_surface_bounds(self):
        """Bounds of all original support samples, beyond the inscribed patch."""
        points = self._measured_surface_tree.data
        bounds = np.stack((points.min(0), points.max(0)))
        return np.frombuffer(bounds.tobytes(), dtype=bounds.dtype).reshape(2, 3)

    def measured_surface_distance(self, point):
        """Distance to original selected depth, without claiming free space.

        Consumers must separately check the plane, footprint and permitted
        contact depth. This supports bounded association across camera views.
        """
        return float(self._measured_surface_tree.query(np.asarray(point, float))[0])

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

    def pickup_observation_bounds(self, footprint, *, approximate=False):
        """Bracket an occluded footprint for a visual resting-height check.

        The selected plane must have nearby measured support outside all four
        footprint corners. The returned region is only a pickup hypothesis.
        Contact admission still uses the original patch and depth rays.
        """
        footprint = bounds_array(footprint)
        margin = self.patch.measurement_margin_m
        xy = footprint[:, :2]
        if np.all(xy[0] >= self.patch.bounds[0, :2]) and np.all(xy[1] <= self.patch.bounds[1, :2]):
            return self.patch.bounds

        y, x = np.nonzero(self.support_mask)
        z = self.depth[y, x]
        k, pose = self.intrinsic, self.world_from_camera
        rays = np.column_stack(((x - k[0, 2]) / k[0, 0], (y - k[1, 2]) / k[1, 1], np.ones(len(x))))
        world = (rays * z[:, None]) @ pose[:3, :3].T + pose[:3, 3]
        nearby_m = .035
        if approximate:
            # A visible support beside a partly occluded object supplies a
            # resting hypothesis; it need not surround every padded box corner.
            delta = np.maximum(np.maximum(xy[0]-world[:, :2], world[:, :2]-xy[1]), 0.)
            if np.count_nonzero(np.linalg.norm(delta, axis=1) <= nearby_m) < 4:
                return None
            # Partial-view extents change as another camera contributes
            # visible surfaces. A 5 mm footprint reserve incorrectly treated
            # that change as loss of support. Keep the resting hypothesis
            # local to the same 35 mm measured-support neighborhood; this
            # does not expand collision/contact admission or placement bounds.
            low, high = xy + np.array([[-.02], [.02]])
            corners = np.array([[low[0], low[1]], [high[0], high[1]]])
            height = corners @ self.patch.plane[:2] + self.patch.plane[2]
            return np.array([[*low, height.min()-margin], [*high, height.max()+margin]])
        for i, j in np.ndindex(2, 2):
            corner = np.array([xy[i, 0], xy[j, 1]])
            outward = (world[:, :2] - corner) * [2 * i - 1, 2 * j - 1]
            bracket = np.all((outward >= margin) & (outward <= nearby_m), axis=1)
            if np.count_nonzero(bracket) < 4:
                return None

        # Give the settling check a local allowance for tracking differences.
        # This volume never becomes a free-space or material witness.
        low, high = xy + np.array([[-.005], [.005]])
        corners = np.array([[low[0], low[1]], [low[0], high[1]], [high[0], low[1]], [high[0], high[1]]])
        height = corners @ self.patch.plane[:2] + self.patch.plane[2]
        return np.array([[*low, height.min() - margin], [*high, height.max() + margin]])

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


@dataclass(frozen=True)
class ObservedProbeBand:
    """A local exception for the underside of the held payload during a probe."""

    support: ObservedSupportContact
    half_height_m: float

    def __post_init__(self):
        if not isinstance(self.support, ObservedSupportContact) or not 0 < self.half_height_m <= MAX_TRAVEL_M / 2:
            raise ValueError("invalid_probe_contact_band")

    def covers_box(self, bounds):
        patch = self.support.patch
        if (
            np.any(bounds[0, :2] < patch.bounds[0, :2] + patch.measurement_margin_m)
            or np.any(bounds[1, :2] > patch.bounds[1, :2] - patch.measurement_margin_m)
        ):
            return False
        normal, offset = self.support.unit_plane
        heights = box_corners(bounds) @ normal - offset
        return bool(heights.min() >= -self.half_height_m and heights.max() <= self.half_height_m)
