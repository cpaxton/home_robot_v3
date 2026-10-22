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
