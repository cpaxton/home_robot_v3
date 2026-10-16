# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Task-local weighted coverage, independent of cameras, models and selection.

Visibility is supplied by the caller (prefer depth-observed world cells). This
module never infers visibility through unknown space. Cell IDs and target XY
must use the same map frame. XY coverage intentionally collapses height.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True)
class CoverageGrid:
    """Stable XY cells from observed world points; no ray filling or dilation.

    Pass depth-valid surface endpoints, not a camera frustum or inferred free
    space. All views must use this same grid and the objective's map scope.
    """

    resolution_m: float
    origin_xy: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        if not math.isfinite(self.resolution_m) or self.resolution_m <= 0:
            raise ValueError("resolution_m must be finite and positive")
        if len(self.origin_xy) != 2 or not all(math.isfinite(v) for v in self.origin_xy):
            raise ValueError("origin_xy must be finite XY")
        object.__setattr__(self, "origin_xy", tuple(self.origin_xy))

    def observed_cells(self, world_xy: np.ndarray) -> frozenset[str]:
        xy = np.asarray(world_xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("observed world XY must have shape (N, 2)")
        xy = xy[np.isfinite(xy).all(axis=1)]
        indices = np.floor((xy - self.origin_xy) / self.resolution_m)
        if np.any(np.abs(indices) >= 2**53):
            raise ValueError("cell indices exceed exact integer representation")
        return frozenset(f"{int(x)},{int(y)}" for x, y in np.unique(indices, axis=0))

    def centers(self, cells) -> dict[str, tuple[float, float]]:
        result = {}
        for cell in sorted(set(cells)):
            x, y = (int(v) for v in cell.split(","))
            result[cell] = (
                self.origin_xy[0] + (x + 0.5) * self.resolution_m,
                self.origin_xy[1] + (y + 0.5) * self.resolution_m,
            )
        return result


def depth_surface_cells(
    depth, intrinsics, camera_to_world, *, grid: CoverageGrid, max_depth_m: float, pixel_stride: int = 4
) -> frozenset[str]:
    """Project valid optical-Z depth endpoints to map XY; never fill rays.

    Depth must be registered to the supplied camera and measured in meters.
    Stride and depth range are explicit experimental knobs, not confidence
    thresholds. The world transform must already use the map's planar XY axes.
    """
    world_xy = _depth_surface_world_xy(
        depth, intrinsics, camera_to_world, max_depth_m=max_depth_m, pixel_stride=pixel_stride
    )
    return grid.observed_cells(world_xy)


def _depth_surface_world_xy(depth, intrinsics, camera_to_world, *, max_depth_m, pixel_stride):
    depth = np.asarray(depth, dtype=float)
    k = np.asarray(intrinsics, dtype=float)
    pose = np.asarray(camera_to_world, dtype=float)
    if depth.ndim != 2 or min(depth.shape) == 0:
        raise ValueError("depth must be a nonempty HxW array")
    if k.shape != (3, 3) or not np.isfinite(k).all() or abs(np.linalg.det(k)) < 1e-12:
        raise ValueError("intrinsics must be a finite invertible 3x3 matrix")
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0, 0, 0, 1])
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
    ):
        raise ValueError("camera_to_world must be a rigid 4x4 transform")
    if not isinstance(pixel_stride, int) or isinstance(pixel_stride, bool) or pixel_stride < 1:
        raise ValueError("pixel_stride must be a positive integer")
    if not math.isfinite(max_depth_m) or max_depth_m <= 0:
        raise ValueError("max_depth_m must be finite and positive")
    yy, xx = np.mgrid[0 : depth.shape[0] : pixel_stride, 0 : depth.shape[1] : pixel_stride]
    z = depth[::pixel_stride, ::pixel_stride].ravel()
    valid = np.isfinite(z) & (z > 0) & (z <= max_depth_m)
    rays = np.stack([xx.ravel(), yy.ravel(), np.ones(z.size)], axis=1)[valid] @ np.linalg.inv(k).T
    points = rays * z[valid, None]
    world = points @ pose[:3, :3].T + pose[:3, 3]
    return world[:, :2]


def depth_surface_quality(
    depth,
    intrinsics,
    camera_to_world,
    *,
    grid: CoverageGrid,
    max_depth_m: float,
    pixel_stride: int = 4,
    reference_pixels: float = 64,
    context_credit: float = 0.5,
) -> dict[str, float]:
    """Pixel-support proxy, NOT calibrated perceptual confidence.

    Quality = context_credit + (1-context_credit)*min(pixel_support/ref, 1).
    Nonzero context credit preserves value for wider views; max quality across
    views prevents double credit. Stride-squared scaling approximates full-image
    support and assumes the same RGB resolution for all candidates. XY collapses
    height, so vertical surfaces can have large support; this is an ablation.
    """
    if not math.isfinite(reference_pixels) or reference_pixels <= 0:
        raise ValueError("reference_pixels must be finite and positive")
    if not math.isfinite(context_credit) or not 0 <= context_credit <= 1:
        raise ValueError("context_credit must be in [0, 1]")
    xy = _depth_surface_world_xy(depth, intrinsics, camera_to_world, max_depth_m=max_depth_m, pixel_stride=pixel_stride)
    indices = np.floor((xy - grid.origin_xy) / grid.resolution_m)
    if np.any(np.abs(indices) >= 2**53):
        raise ValueError("cell indices exceed exact integer representation")
    unique, counts = np.unique(indices, axis=0, return_counts=True)
    return {
        f"{int(x)},{int(y)}": float(
            context_credit + (1 - context_credit) * min(1, n * pixel_stride**2 / reference_pixels)
        )
        for (x, y), n in zip(unique, counts, strict=True)
    }


@dataclass(frozen=True)
class CoverageObjective:
    scope_id: str
    cell_weights: Mapping[str, float]
    visibility_source: str

    def __post_init__(self):
        if not self.scope_id or not self.visibility_source:
            raise ValueError("coverage scope and visibility provenance are required")
        weights = dict(self.cell_weights)
        if any(not isinstance(k, str) or not k or not math.isfinite(v) or v < 0 for k, v in weights.items()):
            raise ValueError("cell weights must have string IDs and finite nonnegative values")
        if not weights or not math.isfinite(math.fsum(weights.values())) or not any(weights.values()):
            raise ValueError("coverage objective must have positive finite weight")
        object.__setattr__(self, "cell_weights", MappingProxyType(weights))

    def weight(self, cells) -> float:
        """Union weight; repeated cells never earn repeated credit."""
        return math.fsum(self.cell_weights.get(cell, 0.0) for cell in sorted(set(cells)))

    def marginal(self, visible, covered) -> float:
        return self.weight(set(visible) - set(covered))

    def quality_marginal(self, quality, best) -> float:
        return math.fsum(
            self.cell_weights.get(cell, 0) * max(0, q - best.get(cell, 0)) for cell, q in sorted(quality.items())
        )


def gaussian_coverage_objective(
    cell_centers_xy: Mapping[str, tuple[float, float]],
    *,
    target_xy: tuple[float, float],
    sigma_m: float,
    scope_id: str,
    visibility_source: str,
) -> CoverageObjective:
    """A Gaussian hill around a supplied target, not around the camera.

    Increase sigma for uncertain target positions. The caller chooses the map
    support/ROI; no answer labels or object classes enter these weights.
    """
    if len(target_xy) != 2 or not all(math.isfinite(x) for x in target_xy):
        raise ValueError("target_xy must be finite XY")
    if not math.isfinite(sigma_m) or sigma_m <= 0:
        raise ValueError("sigma_m must be finite and positive")
    weights = {}
    for cell, xy in cell_centers_xy.items():
        if len(xy) != 2 or not all(math.isfinite(x) for x in xy):
            raise ValueError("cell centers must be finite XY")
        distance = math.hypot(xy[0] - target_xy[0], xy[1] - target_xy[1]) / sigma_m
        weights[cell] = math.exp(-0.5 * distance * distance)
    return CoverageObjective(scope_id, weights, visibility_source)
