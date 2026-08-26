# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""DynaMem localize_text must feed RGB (not OpenCV BGR) to the detector."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from emet.mapping.voxel.voxel_dynamem import SparseVoxelMap


def test_localize_feature_similarity_passes_rgb_not_bgr(tmp_path):
    vm = SparseVoxelMap.__new__(SparseVoxelMap)
    rgb = torch.zeros(32, 40, 3, dtype=torch.uint8)
    rgb[..., 0] = 220
    seen: dict[str, float] = {}

    class _Det:
        def compute_obj_coord(self, text, rgb_in, depth, camera_K, camera_pose, **_k):
            arr = rgb_in.detach().cpu().numpy() if hasattr(rgb_in, "detach") else np.asarray(rgb_in)
            seen["mean_r"] = float(arr[..., 0].mean())
            seen["mean_b"] = float(arr[..., 2].mean())
            return None

    vm.detection_model = _Det()
    vm.log = str(tmp_path)
    vm.observations = [
        SimpleNamespace(
            rgb=rgb,
            camera_pose=torch.eye(4),
            depth=torch.ones(32, 40),
            camera_K=torch.eye(3),
        )
    ]
    points = torch.tensor([[0.0, 0.0, 1.0]])
    vm.semantic_memory = MagicMock()
    vm.semantic_memory.get_pointcloud.return_value = (points, None, None, None)
    vm.semantic_memory._obs_counts = torch.tensor([1])
    vm.find_alignment_over_model = lambda _text: torch.tensor([0.11])

    out = vm.localize_with_feature_similarity("red cylinder", debug=False)
    assert out is None
    assert seen["mean_r"] > 200
    assert seen["mean_b"] < 20
    assert vm._last_localize_stats["yoloe_hit"] is False
    assert vm._last_localize_stats["max_cosine"] == pytest.approx(0.11)
