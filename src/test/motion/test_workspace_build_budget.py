# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Batch bounds preserve free/occupied/unknown geometry and reject before allocation."""

import numpy as np
import pytest

from emet.motion.observed_workspace import ObservedWorkspace


@pytest.mark.parametrize("chunk", [1, 7, 37, 4096])
def test_batch_boundaries_keep_exact_scene_evidence(chunk):
    depth = np.random.default_rng(7).uniform(0.6, 1.2, (31, 43))
    depth[8:12, 15:18] = 0
    depth[14, 22] = np.nan
    pose = np.eye(4)
    pose[:3, 3] = [0.013, -0.027, 0.03]
    kwargs = {"workspace": [[-0.2, -0.2, 0.1], [0.2, 0.2, 1.3]], "resolution": 0.1}
    intrinsic = [[35, 0, 21], [0, 31, 15], [0, 0, 1]]
    reference = ObservedWorkspace.from_depth(depth, intrinsic, pose, **kwargs, chunk_cells=10000)
    stats = {}
    measured = ObservedWorkspace.from_depth(depth, intrinsic, pose, **kwargs, chunk_cells=chunk, build_stats=stats)
    np.testing.assert_array_equal(measured.cells, reference.cells)
    assert stats["max_cells_per_batch"] <= chunk
    assert stats["output_bytes"] == measured.cells.nbytes


@pytest.mark.parametrize("budget", [{"max_cells": 1}, {"max_output_bytes": 1}, {"max_working_bytes": 1}])
def test_budget_rejects_before_allocating_the_grid(monkeypatch, budget):
    def forbidden(*args, **kwargs):
        raise AssertionError("grid allocation must not be attempted")

    monkeypatch.setattr(np, "zeros", forbidden)
    with pytest.raises(ValueError, match="observed_workspace_budget_exceeded"):
        ObservedWorkspace.from_depth(
            np.ones((9, 9)), np.eye(3), np.eye(4), workspace=[[0, 0, 0], [1, 1, 1]], resolution=0.1, **budget
        )


def test_tiny_working_budget_rejects_before_depth_conversion_or_sanitation(monkeypatch):
    depth = np.ones((100, 100), dtype=np.float32)
    original = np.asarray

    def checked(value, dtype=None, *args, **kwargs):
        if value is depth and dtype is not None:
            raise AssertionError("depth conversion entered before budget rejection")
        return original(value, dtype, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("full-frame sanitation entered before budget rejection")

    monkeypatch.setattr(np, "asarray", checked)
    monkeypatch.setattr(np, "where", forbidden)
    with pytest.raises(ValueError, match="observed_workspace_budget_exceeded"):
        ObservedWorkspace.from_depth(
            depth, np.eye(3), np.eye(4), workspace=[[0, 0, 0], [1, 1, 1]], resolution=0.1, max_working_bytes=1
        )
