# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Guard empty point clouds in list_objects_in_an_image (navigate/update crash)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from emet.mapping.voxel.voxel_dynamem import SparseVoxelMap


def test_list_objects_in_an_image_handles_empty_cloud():
    vm = SparseVoxelMap.__new__(SparseVoxelMap)
    vm.image_description_client = MagicMock()
    vm.image_descriptions = []
    vm.voxel_pcd = SimpleNamespace(
        _obs_counts=torch.zeros(0, dtype=torch.long),
        get_pointcloud=lambda: (torch.zeros(0, 3), None, None, None),
    )
    vm.xy_to_grid_coords = MagicMock(return_value=[0, 0])

    # Force VLM path to return empty so we exercise the cloud branch.
    import emet.mapping.voxel.dynamem_eqa as mod

    orig = mod.dynamem_vllm_call
    mod.dynamem_vllm_call = lambda *a, **k: ""
    try:
        vm.list_objects_in_an_image(torch.zeros(8, 8, 3, dtype=torch.uint8).numpy())
    finally:
        mod.dynamem_vllm_call = orig

    assert len(vm.image_descriptions) == 1
    assert vm.image_descriptions[0][1] == [0, 0]
