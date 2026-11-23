# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
import numpy as np

from emet.motion.depth_surface_validity import supported_depth


def test_isolated_depth_is_unqualified():
    depth = np.ones((41, 41))
    depth[20, 20] = .5
    assert not supported_depth(depth)[20, 20]


def test_two_pixel_thin_surface_is_retained():
    depth = np.ones((10, 10))
    depth[4:6, 4] = .5
    assert supported_depth(depth)[4:6, 4].all()
    assert supported_depth(depth).all()
