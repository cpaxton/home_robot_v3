# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from types import SimpleNamespace

import pytest

from emet.controller.controller_lazy_graph import LazyGraphController


@pytest.mark.parametrize("backend,strategy", [("sam2", "point"), ("rgbd", "depth_candidates")])
def test_invalid_tracking_configuration_fails_before_model_or_segmenter_setup(backend, strategy):
    controller = SimpleNamespace(
        parameters={
            "query_memory": {
                "mask_backend": backend,
                "region_strategy": strategy,
                "track_grounded_box": True,
            }
        }
    )
    with pytest.raises(ValueError, match="query_memory.region_strategy=depth_candidates"):
        LazyGraphController.ground_vlm_frame(controller, None, "cup", "cup", tracking_target=object())


def test_rgbd_search_padding_reaches_existing_grounding_entrypoint(monkeypatch):
    from unittest.mock import Mock

    import emet.memory.vlm_region_grounding as grounding

    result = (None, [], [], {"valid": False})
    ground = Mock(return_value=result)
    monkeypatch.setattr(grounding, "ground_vlm_region", ground)
    controller = SimpleNamespace(
        parameters={
            "query_memory": {"mask_backend": "rgbd", "region_strategy": "depth_candidates", "search_padding": 50}
        },
        graph_memory=SimpleNamespace(eqa_client=object()),
        voxel_map=SimpleNamespace(min_depth=0.2, max_depth=4.0),
    )
    assert LazyGraphController.ground_vlm_frame(controller, None, "cup", "cup") is result
    assert ground.call_args.kwargs["search_padding"] == 50
    assert ground.call_args.kwargs["strategy"] == "depth_candidates"
    assert "segmenter" not in ground.call_args.kwargs


def test_normal_split_forwards_measured_world_geometry(monkeypatch):
    from unittest.mock import Mock

    import numpy as np

    import emet.memory.vlm_region_grounding as grounding

    ground = Mock(return_value=(None, [], [], {"valid": False}))
    monkeypatch.setattr(grounding, "ground_vlm_region", ground)
    controller = SimpleNamespace(
        parameters={
            "query_memory": {
                "mask_backend": "rgbd",
                "region_strategy": "depth_candidates",
                "split_surface_normals": True,
            }
        },
        graph_memory=SimpleNamespace(eqa_client=object()),
        voxel_map=SimpleNamespace(min_depth=0.2, max_depth=4.0),
    )
    frame = SimpleNamespace(depth=np.ones((40, 40)), full_world_xyz=np.ones((40, 40, 3)))
    LazyGraphController.ground_vlm_frame(controller, frame, "cup", "cup")
    assert ground.call_args.kwargs["world_xyz"].shape == (40, 40, 3)
    with pytest.raises(ValueError, match="requires world-aligned depth"):
        LazyGraphController.ground_vlm_frame(controller, None, "cup", "cup")
