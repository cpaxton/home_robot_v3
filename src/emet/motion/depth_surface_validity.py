# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Reject unsupported isolated depth returns while retaining them as unknown rays."""
import numpy as np


def supported_depth(depth, *, jump_m=.02):
    depth = np.asarray(depth)
    valid = np.isfinite(depth) & (depth >= .1) & (depth <= 5.)
    supported = np.zeros(depth.shape, dtype=bool)
    for axis in (0, 1):
        first, second = [slice(None)] * 2, [slice(None)] * 2
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        first, second = tuple(first), tuple(second)
        agrees = valid[first] & valid[second] & (np.abs(depth[first] - depth[second]) <= jump_m)
        supported[first] |= agrees
        supported[second] |= agrees
    return valid & supported
