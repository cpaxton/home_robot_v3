# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Bounded RGBD reassociation of an already selected visible surface.

This provides current visible points, not object pose, complete volume, contact,
or long-term re-identification. Labels must come from the reference observation;
geometry and appearance can reject an association but cannot name a new object.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from emet.memory.surface_candidates import candidate_mask, surface_candidates


@dataclass(frozen=True)
class ReacquiredSurface:
    mask: np.ndarray
    points: np.ndarray
    reference_support_fraction: float


def reacquire_surface(
    reference_rgb,
    reference_world_xyz,
    reference_mask,
    rgb,
    depth,
    world_xyz,
    *,
    min_depth,
    max_depth,
    distance_m=0.025,
    color_distance=35.0,
    diagnostics=None,
):
    """Require one whole measured component near the fixed reference surface.

    Matching always uses the original reference, never an incrementally drifting
    centroid. A nearby candidate must preserve both reference coverage and its
    own support, appearance, and surface extent. Ambiguity rejects the complete
    update. Search bounds are untrusted; a component clipped by the search crop
    is rejected rather than trimmed into an apparent target.
    """
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics["candidates"] = []
    reference_rgb, rgb = np.asarray(reference_rgb), np.asarray(rgb)
    depth, world_xyz = np.asarray(depth), np.asarray(world_xyz)
    reference_world_xyz, reference_mask = np.asarray(reference_world_xyz), np.asarray(reference_mask)
    if (
        depth.ndim != 2
        or rgb.shape != (*depth.shape, 3)
        or rgb.dtype != np.uint8
        or world_xyz.shape != (*depth.shape, 3)
        or reference_mask.dtype != bool
        or reference_mask.ndim != 2
        or reference_rgb.shape != (*reference_mask.shape, 3)
        or reference_rgb.dtype != np.uint8
        or reference_world_xyz.shape != (*reference_mask.shape, 3)
        or not np.isfinite(distance_m)
        or not 0 < distance_m <= 0.05
        or not np.isfinite(color_distance)
        or not 0 < color_distance <= 50
    ):
        raise ValueError("invalid_surface_reacquisition_inputs")
    ref_valid = reference_mask & np.isfinite(reference_world_xyz).all(axis=-1)
    if ref_valid.sum() < 25 or ref_valid.sum() != reference_mask.sum():
        raise ValueError("invalid_reference_surface")
    reference = reference_world_xyz[ref_valid]
    colors = reference_rgb[ref_valid].astype(float)
    valid = np.isfinite(world_xyz).all(axis=-1) & np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
    near = valid & (world_xyz >= reference.min(axis=0) - distance_m).all(axis=-1)
    near &= (world_xyz <= reference.max(axis=0) + distance_m).all(axis=-1)
    diagnostics["near_pixels"] = int(near.sum())
    diagnostics["reference_center"] = np.median(reference, axis=0).tolist()
    if near.sum() < 25:
        raise ValueError("surface_not_reacquired")
    ys, xs = np.where(near)
    h, w = depth.shape
    # Include surrounding context so support furniture remains a whole competing
    # surface, rather than becoming a small patch clipped to the old object box.
    lo = np.maximum([xs.min() - 10, ys.min() - 10], [0, 0])
    hi = np.minimum([xs.max() + 11, ys.max() + 11], [w, h])
    box = [
        int(np.floor(lo[0] * 1000 / w)),
        int(np.floor(lo[1] * 1000 / h)),
        int(np.ceil(hi[0] * 1000 / w)),
        int(np.ceil(hi[1] * 1000 / h)),
    ]
    regions = surface_candidates(depth, box, min_depth=min_depth, max_depth=max_depth, rgb=rgb, world_xyz=world_xyz)
    tree = cKDTree(reference)
    matches = []
    for region in regions:
        mask = candidate_mask(region, depth.shape)
        entry = {"id": region["id"], "pixels": int(mask.sum())}
        diagnostics["candidates"].append(entry)
        y, x = np.where(mask)
        # Edges at the camera boundary are also incomplete evidence.
        if x.min() <= lo[0] or y.min() <= lo[1] or x.max() >= hi[0] - 1 or y.max() >= hi[1] - 1:
            entry["rejected"] = "search_boundary"
            continue
        if not valid[mask].all():
            entry["rejected"] = "invalid_depth"
            continue
        points = world_xyz[mask]
        entry["center"] = np.median(points, axis=0).tolist()
        if not 0.35 <= len(points) / len(reference) <= 3.0:
            entry["rejected"] = "size_change"
            continue
        if np.max(np.abs(np.median(rgb[mask].astype(float), axis=0) - np.median(colors, axis=0))) > color_distance:
            entry["rejected"] = "appearance_change"
            continue
        current_support = np.mean(tree.query(points)[0] <= distance_m)
        reference_support = np.mean(cKDTree(points).query(reference)[0] <= distance_m)
        entry.update(current_support=float(current_support), reference_support=float(reference_support))
        if current_support >= 0.9 and reference_support >= 0.8:
            matches.append(ReacquiredSurface(mask.copy(), points.copy(), float(reference_support)))
    if len(matches) != 1:
        raise ValueError("surface_not_reacquired" if not matches else "surface_association_ambiguous")
    return matches[0]
