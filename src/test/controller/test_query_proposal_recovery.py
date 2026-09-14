# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.controller.controller_lazy_graph import LazyGraphController


def test_recovery_is_enabled_only_in_experimental_preset():
    from emet.core.parameters import get_parameters

    candidate = get_parameters("configs/emet/query_geometry_recovery_pilot.yaml")
    assert candidate.get("query_memory")["recover_proposals_with_vlm"] is True
    for name in (
        "query_detector_segmented_pilot",
        "query_segmented_support_pilot",
        "query_geometry_contact_aperture_pilot",
    ):
        control = get_parameters(f"configs/emet/{name}.yaml")
        assert not control.get("query_memory").get("recover_proposals_with_vlm", False)
    control = get_parameters("configs/emet/query_geometry_contact_aperture_pilot.yaml")
    candidate.data["query_memory"].pop("recover_proposals_with_vlm")
    assert candidate.data == control.data


def test_tracking_preset_only_changes_proposal_source_for_known_targets():
    from emet.core.parameters import get_parameters

    candidate = get_parameters("configs/emet/query_geometry_tracked_pilot.yaml")
    control = get_parameters("configs/emet/query_geometry_setdown_pilot.yaml")
    assert candidate.data["query_memory"].pop("track_grounded_box") is True
    assert candidate.data == control.data


@pytest.mark.parametrize("valid,kind", [(True, None), (False, None), (False, "candidate_overflow")])
def test_projected_bounds_are_only_proposals_and_do_not_retry_rejection(monkeypatch, valid, kind):
    detector = Mock(side_effect=AssertionError("Known-target tracking must not reload detector proposals"))
    monkeypatch.setattr("emet.perception.detection.yoloe.get_shared_yoloe_perception", detector)
    segmenter = Mock()
    segmenter.segment.return_value = np.ones((1, 10, 10), dtype=bool)
    target = SimpleNamespace(
        candidate_id=8,
        observation_revision=9,
        project_box=Mock(return_value=np.array([1.0, 2.0, 8.0, 9.0])),
        select_mask=Mock(),
    )
    audit = {"valid": valid, "failure_kind": kind}
    ground = Mock(return_value=(None, [], [], audit))
    monkeypatch.setattr("emet.memory.vlm_region_grounding.ground_vlm_region", ground)
    controller = SimpleNamespace(
        graph_memory=SimpleNamespace(eqa_client=Mock()),
        voxel_map=SimpleNamespace(min_depth=0.25, max_depth=2.5),
        parameters={
            "query_memory": {
                "mask_backend": "yoloe_sam2",
                "track_grounded_box": True,
                "recover_proposals_with_vlm": True,
            }
        },
        device="cpu",
        _query_segmenter=segmenter,
    )
    frame = SimpleNamespace(
        rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        depth=np.ones((10, 10)),
        camera_K=np.eye(3),
        camera_pose=np.eye(4),
        full_world_xyz=np.ones((10, 10, 3)),
    )
    result = LazyGraphController.ground_vlm_frame(controller, frame, "cup", "cup", tracking_target=target)
    ground.assert_called_once()
    assert ground.call_args.kwargs["proposal_masks"] is segmenter.segment.return_value
    assert "segmenter" not in ground.call_args.kwargs
    np.testing.assert_array_equal(segmenter.segment.call_args.args[1], [[1, 2, 8, 9]])
    assert result[3]["valid"] is valid
    assert result[3]["tracking_proposal"]["observation_revision"] == 9
    assert "proposal_recovery" not in result[3]
    # The pre-selection predicate uses the same whole-mask association as the
    # post-selection guard, without deleting pixels to manufacture a match.
    mask = np.ones((10, 10), dtype=bool)
    predicate = ground.call_args.kwargs["candidate_filter"]
    assert predicate(mask) is True
    target.select_mask.assert_called_once()
    np.testing.assert_array_equal(target.select_mask.call_args.args[1], mask)
    np.testing.assert_array_equal(target.select_mask.call_args.args[2], frame.full_world_xyz)
    target.select_mask.side_effect = ValueError("wrong object")
    assert predicate(mask) is False
    frame.full_world_xyz = None
    with pytest.raises(ValueError, match="world-aligned depth"):
        predicate(mask)


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
    ignored_target = Mock()
    result = LazyGraphController.ground_vlm_frame(
        controller, frame, "cup", "cup", min_depth=0.0, tracking_target=ignored_target
    )
    ignored_target.project_box.assert_not_called()
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
