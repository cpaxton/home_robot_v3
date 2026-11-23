# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
import numpy as np

from emet.motion.surface_contact_patches import SurfaceContactPatches


def plane():
    x, y = np.meshgrid(np.linspace(-.025, .025, 31), np.linspace(-.025, .025, 31))
    return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))


def test_patch_accepts_new_samples_along_surface_but_not_other_surface_or_unbounded_sliding():
    patches = SurfaceContactPatches([[0., 0., 0.]], plane())
    assert patches.assign([[.02, .01, .0005]]).tolist() == [0]
    assert patches.assign([[.02, .01, .012], [.04, 0., 0.]]).tolist() == [-1, -1]
    # Association never grows to follow the robot along the surface.
    patches.assign([[.029, 0., 0.]])
    assert patches.assign([[.04, 0., 0.]]).tolist() == [-1]


def test_sparse_or_collinear_evidence_cannot_explain_a_large_plane():
    patches = SurfaceContactPatches([[0., 0., 0.]], [[0., 0., 0.], [.001, 0., 0.]])
    assert patches.assign([[.001, 0., 0.], [.01, 0., 0.]]).tolist() == [0, -1]
    points = np.column_stack((np.linspace(-.01, .01, 30), np.zeros((30, 2))))
    assert SurfaceContactPatches([[0., 0., 0.]], points).assign([[0., .01, 0.]]).tolist() == [-1]


def test_patch_membership_is_stable_when_surface_samples_are_reordered():
    seeds = np.array([[.02, 0., 0.], [-.02, 0., 0.]])
    a = SurfaceContactPatches(seeds, plane())
    b = SurfaceContactPatches(seeds[::-1], plane()[::-1])
    np.testing.assert_array_equal(a.assign(plane()), b.assign(plane()))
