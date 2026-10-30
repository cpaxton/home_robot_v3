# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Measured planar support patches with a fully observed rectangular footprint."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emet.controller.manipulation.native_inputs import ObservationUnavailable


def _largest_rectangle(cells):
    heights = np.zeros(cells.shape[1], int)
    best, area = None, 0
    for row in range(cells.shape[0]):
        heights = np.where(cells[row], heights + 1, 0)
        stack = []
        for col in range(cells.shape[1] + 1):
            height = int(heights[col]) if col < cells.shape[1] else 0
            start = col
            while stack and stack[-1][1] > height:
                left, value = stack.pop()
                if value * (col - left) > area:
                    best, area = (row - value + 1, left, row + 1, col), value * (col - left)
                start = left
            if not stack or stack[-1][1] < height:
                stack.append((start, height))
    return best


@dataclass(frozen=True)
class ObservedSupportPatch:
    bounds: np.ndarray
    plane: np.ndarray  # z = a*x + b*y + c
    residual_m: float
    observation_id: str
    received_monotonic: float
    measurement_margin_m: float

    def __post_init__(self):
        bounds, plane = np.asarray(self.bounds, float), np.asarray(self.plane, float)
        if (
            bounds.shape != (2, 3)
            or plane.shape != (3,)
            or not np.isfinite(bounds).all()
            or not np.isfinite(plane).all()
            or np.any(bounds[0] >= bounds[1])
            or not self.observation_id
            or not np.isfinite(self.received_monotonic)
            or not np.isfinite([self.residual_m, self.measurement_margin_m]).all()
            or not 0 <= self.residual_m <= self.measurement_margin_m
            or self.measurement_margin_m <= 0
        ):
            raise ValueError("invalid_support_patch_evidence")
        for name in ("bounds", "plane"):
            value = np.asarray(getattr(self, name), float)
            object.__setattr__(self, name, np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape))


def fit_observed_support(
    observation, mask, *, observation_id, measurement_margin_m, resolution=0.01, min_extent_m=0.05, max_cells=20_000
):
    """Fit a selected support instance; certify cell projections, not its AABB.

    The caller supplies semantic mask evidence and independently calibrated
    measurement margin. No hole filling or whole-receptacle shape is inferred.
    """
    from emet.controller.task.tamp.observed_capture import capture_arrays

    _, depth, world = capture_arrays(observation)
    mask = np.asarray(mask)
    if (
        mask.dtype != bool
        or mask.shape != depth.shape
        or mask.sum() < 25
        or not np.isfinite(measurement_margin_m)
        or not 0 < measurement_margin_m <= 0.02
        or not np.isfinite(resolution)
        or resolution <= 0
        or not observation_id
    ):
        raise ObservationUnavailable("support_geometry_unknown")
    valid = np.isfinite(depth) & (depth > 0.1) & np.isfinite(world).all(-1)
    if not valid[mask].all():
        raise ObservationUnavailable("support_geometry_unknown")
    points = world[mask]
    design = np.column_stack((points[:, :2], np.ones(len(points))))
    plane, _, rank, _ = np.linalg.lstsq(design, points[:, 2], rcond=None)
    residual = float(np.max(np.abs(design @ plane - points[:, 2])))
    if rank < 3 or np.linalg.norm(plane[:2]) > np.tan(np.deg2rad(5)) or residual > measurement_margin_m:
        raise ObservationUnavailable("support_plane_unqualified")
    origin = np.ceil(points[:, :2].min(0) / resolution) * resolution
    shape = np.floor((points[:, :2].max(0) - origin) / resolution).astype(int)
    if np.any(shape < 1) or np.prod(shape) > max_cells:
        raise ObservationUnavailable("support_patch_unavailable")
    known = np.zeros(tuple(shape), bool)
    pose = np.asarray(observation.camera_pose)
    k = np.asarray(observation.camera_K)
    for x, y in np.ndindex(*shape):
        xy = origin + (np.array([[x, y], [x + 1, y], [x, y + 1], [x + 1, y + 1]])) * resolution
        corners = np.column_stack((xy, xy @ plane[:2] + plane[2]))
        camera = (corners - pose[:3, 3]) @ pose[:3, :3]
        if np.any(camera[:, 2] <= 0):
            continue
        projected = camera @ k.T
        uv = projected[:, :2] / projected[:, 2:]
        lo, hi = np.floor(uv.min(0)).astype(int), np.ceil(uv.max(0)).astype(int)
        if np.any(lo < 0) or hi[0] >= mask.shape[1] or hi[1] >= mask.shape[0]:
            continue
        patch = mask[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
        known[x, y] = bool(patch.size and patch.all())
    rectangle = _largest_rectangle(known)
    if rectangle is None:
        raise ObservationUnavailable("support_patch_unavailable")
    x0, y0, x1, y1 = rectangle
    lo, hi = origin + np.array([x0, y0]) * resolution, origin + np.array([x1, y1]) * resolution
    if np.any(hi - lo < min_extent_m):
        raise ObservationUnavailable("support_patch_unavailable")
    corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])
    zs = corners @ plane[:2] + plane[2]
    return ObservedSupportPatch(
        np.array([[*lo, zs.min() - measurement_margin_m], [*hi, zs.max() + measurement_margin_m]]),
        plane,
        residual,
        observation_id,
        observation.received_monotonic,
        measurement_margin_m,
    )


def refresh_support_patch(observation, reference, *, observation_id, reference_color=None):
    """Reobserve only the old measured footprint; never renew cached geometry.

    A selected support patch may be a partial table view. Requiring a whole
    connected furniture component here would wrongly reject camera clipping.
    Missing/occluding pixels are excluded from the new measured footprint.
    """
    from emet.controller.task.tamp.observed_capture import capture_arrays

    rgb, depth, world = capture_arrays(observation)
    mask = np.isfinite(depth) & (depth > 0.1) & np.isfinite(world).all(-1)
    mask &= np.all(world[..., :2] >= reference.bounds[0, :2], axis=-1)
    mask &= np.all(world[..., :2] <= reference.bounds[1, :2], axis=-1)
    residual = np.abs(world[..., 2] - (world[..., :2] @ reference.plane[:2] + reference.plane[2]))
    mask &= residual <= reference.measurement_margin_m
    if reference_color is not None:
        mask &= np.max(np.abs(rgb.astype(float) - reference_color), axis=-1) <= 35
    return fit_observed_support(
        observation, mask, observation_id=observation_id, measurement_margin_m=reference.measurement_margin_m
    )


def support_mask_from_seed(observation, selection, *, measurement_margin_m):
    """Grow a measured planar component from an explicitly selected support pixel.

    This is a separate partial-support-patch producer, not an overflow fallback
    for object proposals. It never selects a larger surface elsewhere in view.
    """
    from scipy.ndimage import label

    from emet.controller.task.tamp.observed_capture import capture_arrays

    rgb, depth, world = capture_arrays(observation)
    box, point = selection.get("box"), selection.get("point")
    if (
        selection.get("verified") is not True
        or not isinstance(box, list)
        or len(box) != 4
        or not isinstance(point, list)
        or len(point) != 2
        or any(type(v) is not int or not 0 <= v <= 1000 for v in box + point)
    ):
        raise ObservationUnavailable("support_seed_unavailable")
    h, w = depth.shape
    x0, y0 = np.floor(np.array(box[:2]) * [w, h] / 1000).astype(int)
    x1, y1 = np.ceil(np.array(box[2:]) * [w, h] / 1000).astype(int)
    x, y = min(w - 1, point[0] * w // 1000), min(h - 1, point[1] * h // 1000)
    if not (0 <= x0 <= x < x1 <= w and 0 <= y0 <= y < y1 <= h):
        raise ObservationUnavailable("support_seed_unavailable")
    valid = np.isfinite(depth) & (depth > 0.1) & np.isfinite(world).all(-1)
    if not valid[y, x]:
        raise ObservationUnavailable("support_seed_unavailable")
    appearance = np.max(np.abs(rgb.astype(float) - rgb[y, x].astype(float)), axis=-1) <= 35
    local = np.zeros((h, w), bool)
    local[max(0, y - 8) : min(h, y + 9), max(0, x - 8) : min(w, x + 9)] = True
    local &= valid & appearance & (np.linalg.norm(world - world[y, x], axis=-1) <= 0.05)
    points = world[local]
    if len(points) < 25:
        raise ObservationUnavailable("support_seed_unavailable")
    design = np.column_stack((points[:, :2], np.ones(len(points))))
    plane, _, rank, _ = np.linalg.lstsq(design, points[:, 2], rcond=None)
    if rank < 3 or np.linalg.norm(plane[:2]) > np.tan(np.deg2rad(5)):
        raise ObservationUnavailable("support_plane_unqualified")
    mask = np.zeros((h, w), bool)
    mask[y0:y1, x0:x1] = True
    mask &= valid & appearance
    mask &= np.abs(world[..., 2] - (world[..., :2] @ plane[:2] + plane[2])) <= measurement_margin_m
    components, _ = label(mask)
    component = components[y, x]
    if component == 0:
        raise ObservationUnavailable("support_seed_unavailable")
    return components == component
