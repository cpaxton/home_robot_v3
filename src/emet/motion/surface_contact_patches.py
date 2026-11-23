# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Fixed, bounded measured-surface patches for a single contact departure.

Patches never grow along a search branch. Reobserved samples may join an original
patch only within its tangent footprint and measured normal uncertainty band.
Sparse/nonplanar evidence keeps point-sized association rather than inventing a
plane. No scene labels or simulator geometry are used.
"""

import numpy as np
from scipy.spatial import cKDTree


class SurfaceContactPatches:
    def __init__(self, seeds, surface_points, *, radius_m=.03, normal_band_m=.006, surface_tree=None):
        seeds, points = np.asarray(seeds, float).reshape(-1, 3), np.asarray(surface_points, float).reshape(-1, 3)
        if (not np.isfinite(seeds).all() or not np.isfinite(points).all()
                or not np.isfinite([radius_m, normal_band_m]).all()
                or not 0 < radius_m <= .05 or not 0 <= normal_band_m <= .01):
            raise ValueError('invalid_surface_contact_patch')
        self.radius, self.band = float(radius_m), float(normal_band_m)
        self.anchors, self.normals = [], []
        tree = cKDTree(points) if surface_tree is None else surface_tree
        for seed in seeds[np.lexsort(seeds.T[::-1])]:
            if self.assign(seed[None])[0] >= 0:
                continue
            ids = tree.query_ball_point(seed, min(.015, self.radius))
            normal = None
            if len(ids) >= 6:
                neighborhood = points[ids]
                centered = neighborhood-neighborhood.mean(0)
                values, vectors = np.linalg.eigh(centered.T @ centered / len(ids))
                if values[1] > 1e-7 and values[0] <= .05 * values.sum():
                    normal = vectors[:, 0]
            self.anchors.append(seed.copy())
            self.normals.append(normal)

    def assign(self, points):
        points = np.asarray(points, float).reshape(-1, 3)
        result = np.full(len(points), -1, dtype=int)
        for index, (anchor, normal) in enumerate(zip(self.anchors, self.normals, strict=True)):
            delta = points-anchor
            if normal is None:
                belongs = np.linalg.norm(delta, axis=1) <= min(self.radius, max(.002, self.band))
            else:
                height = delta @ normal
                tangent = delta-height[:, None]*normal
                belongs = (np.abs(height) <= self.band) & (np.linalg.norm(tangent, axis=1) <= self.radius)
            result[(result < 0) & belongs] = index
        return result
