# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Bounded free-region search on explicitly grounded horizontal support patches."""
from dataclasses import dataclass, field

import numpy as np

from emet.motion.placement_geometry import bounds_array


@dataclass
class SurfaceSearchResult:
    centers: list = field(default_factory=list)
    blocker_bounds: list = field(default_factory=list)
    budget_exhausted: bool = False


def support_patches(value):
    """Accept one legacy rectangular patch or a collection of grounded patches."""
    patches = np.asarray(value, dtype=float)
    if patches.shape == (2, 3):
        patches = patches[None]
    if patches.ndim != 3 or patches.shape[1:] != (2, 3) or not len(patches):
        raise ValueError("Explicit support patches required")
    return np.stack([bounds_array(patch) for patch in patches])


def _subtract(rect, obstacle):
    lo, hi = rect
    cut_lo, cut_hi = np.maximum(lo, obstacle[0]), np.minimum(hi, obstacle[1])
    if np.any(cut_lo >= cut_hi):
        return [rect]
    pieces = [
        np.array([lo, [cut_lo[0], hi[1]]]),
        np.array([[cut_hi[0], lo[1]], hi]),
        np.array([[cut_lo[0], lo[1]], [cut_hi[0], cut_lo[1]]]),
        np.array([[cut_lo[0], cut_hi[1]], [cut_hi[0], hi[1]]]),
    ]
    return [piece for piece in pieces if np.all(piece[1] - piece[0] > 1e-9)]


def free_surface_centers(surfaces, *, scene, payload, ee_rotation, clearance_m=.02,
                         preplace_height_m=.12, margin_m=.005, max_centers=64, max_regions=256):
    """Subtract obstacle footprints expanded by the held object's full footprint.

    Obstacles intersecting the entire vertical approach column are retained.
    This finds off-grid free regions, not only a fixed lattice of sample points.
    Robot reach and arm collision remain mandatory downstream checks. Unknown
    space predicates are evaluated on each complete payload approach volume.
    """
    if not 1 <= max_centers <= 64 or not 1 <= max_regions <= 4096:
        raise ValueError("Invalid surface search budget")
    if (not np.isfinite([clearance_m, preplace_height_m, margin_m]).all()
            or clearance_m <= 0 or preplace_height_m <= 0 or margin_m < 0):
        raise ValueError("Invalid placement clearances")
    rotation = np.asarray(ee_rotation, dtype=float).reshape(3, 3)
    if (not np.isfinite(rotation).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1)):
        raise ValueError("Rigid EE rotation required")
    vertices = payload.vertices_ee @ rotation.T
    half = np.ptp(vertices, axis=0) / 2
    result = SurfaceSearchResult()
    proposals = []
    for patch in support_patches(surfaces):
        low = patch[0, :2] + half[:2] + clearance_m
        high = patch[1, :2] - half[:2] - clearance_m
        if np.any(low >= high):
            continue
        z = patch[1, 2] + half[2] + clearance_m
        bottom, top = z - half[2] - margin_m, z + half[2] + preplace_height_m + margin_m
        boxes = scene.boxes
        relevant = ((boxes[:, 1, 2] >= bottom) & (boxes[:, 0, 2] <= top)
                    & np.all(boxes[:, 1, :2] + half[:2] + margin_m >= low, axis=1)
                    & np.all(boxes[:, 0, :2] - half[:2] - margin_m <= high, axis=1))
        regions = [np.stack((low, high))]
        for box in boxes[relevant]:
            result.blocker_bounds.append(box.tolist())
            forbidden = np.stack((box[0, :2] - half[:2] - margin_m - 1e-6,
                                  box[1, :2] + half[:2] + margin_m + 1e-6))
            regions = [piece for region in regions for piece in _subtract(region, forbidden)]
            if len(regions) > max_regions:
                result.budget_exhausted = True
                regions.sort(key=lambda r: -float(np.prod(r[1] - r[0])))
                regions = regions[:max_regions]
            if not regions:
                break
        for region in regions:
            area = float(np.prod(region[1] - region[0]))
            for fraction in ([.5, .5], [.25, .25], [.25, .75], [.75, .25], [.75, .75]):
                xy = region[0] + np.asarray(fraction) * (region[1] - region[0])
                column = np.stack((np.r_[xy - half[:2] - margin_m, bottom],
                                   np.r_[xy + half[:2] + margin_m, top]))
                if not scene.collides(column):
                    proposals.append((area, np.r_[xy, z]))
    proposals.sort(key=lambda item: -item[0])
    seen = set()
    for _, center in proposals:
        key = tuple(np.round(center, 10))
        if key not in seen:
            seen.add(key)
            result.centers.append(center)
            if len(result.centers) >= max_centers:
                break
    return result
