# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.controller.controller_lazy_graph import LazyGraphController


def test_recovery_is_enabled_only_in_experimental_preset():
    from emet.core.parameters import get_parameters

    candidate = get_parameters("configs/emet/query_geometry_contact_aperture_pilot.yaml")
    assert candidate.get("query_memory")["recover_proposals_with_vlm"] is True
    for name in ("query_detector_segmented_pilot", "query_segmented_support_pilot"):
        control = get_parameters(f"configs/emet/{name}.yaml")
        assert not control.get("query_memory").get("recover_proposals_with_vlm", False)


@pytest.mark.parametrize(
    "kind,enabled,recover",
    [
        ("candidate_overflow", True, True),
        ("no_supported_surfaces", True, True),
        ("candidate_overflow", False, False),
        ("invalid_geometry", True, False),
        (None, True, False),
    ],
)
def test_recovery_is_one_alternative_proposal_not_a_semantic_retry(monkeypatch, kind, enabled, recover):
    detector = Mock()
    detector.predict.return_value = (None, np.zeros((10, 10), dtype=int), None)
    monkeypatch.setattr("emet.perception.detection.yoloe.get_shared_yoloe_perception", lambda **kwargs: detector)
    segmenter = Mock()
    segmenter.segment.return_value = np.ones((1, 10, 10), dtype=bool)
    initial = {"valid": False, "reason": "initial rejection"}
    if kind:
        initial["failure_kind"] = kind
    # Even an unsuccessful recovery must not recurse or increase its budget.
    ground = Mock(
        side_effect=[(None, [], [], initial), (None, [], [], {"valid": False, "failure_kind": "candidate_overflow"})]
    )
    monkeypatch.setattr("emet.memory.vlm_region_grounding.ground_vlm_region", ground)
    controller = SimpleNamespace(
        graph_memory=SimpleNamespace(eqa_client=Mock()),
        voxel_map=SimpleNamespace(min_depth=0.25, max_depth=2.5),
        parameters={"query_memory": {"mask_backend": "yoloe_sam2", "recover_proposals_with_vlm": enabled}},
        device="cpu",
        _query_segmenter=segmenter,
    )
    frame = SimpleNamespace(rgb=np.zeros((10, 10, 3), dtype=np.uint8))
    result = LazyGraphController.ground_vlm_frame(controller, frame, "cup", "cup", min_depth=0.0)
    assert ground.call_count == (2 if recover else 1)
    assert not result[3]["valid"]
    if recover:
        first, second = ground.call_args_list
        assert first.args == second.args == (frame, "cup", "cup")
        assert "proposal_masks" in first.kwargs and "segmenter" not in first.kwargs
        assert "proposal_masks" not in second.kwargs and second.kwargs["segmenter"] is segmenter
        assert second.kwargs["min_depth"] == 0.0 and second.kwargs["max_depth"] == 2.5
        assert result[3]["proposal_recovery"]["initial_attempt"] is initial
    else:
        assert "proposal_recovery" not in result[3]
