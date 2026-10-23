# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.memory.vlm_region_grounding import ground_vlm_region, region_depth_mask


def test_depth_support_excludes_background_and_disconnected_regions():
    depth = np.full((40, 40), 3.0)
    depth[10:30, 10:30] = 1.0
    depth[0:5, 0:5] = 1.0
    mask = region_depth_mask(depth, [0, 0, 1000, 1000], [500, 500], min_depth=0.2, max_depth=4)
    assert (mask == 0).sum() == 400
    assert mask[1, 1] == -1


@pytest.mark.parametrize(
    "box,point", [([0, 0, 1000, 1000], [2000, 500]), ([0, 0, 100, 100], [500, 500]), ([0, 0, 1000, 1000], [True, 500])]
)
def test_invalid_region_fails_closed(box, point):
    with pytest.raises(ValueError):
        region_depth_mask(np.ones((40, 40)), box, point, min_depth=0.2, max_depth=4)


def test_vlm_grounding_has_no_detector_dependency():
    frame = SimpleNamespace(
        rgb=np.zeros((40, 40, 3), dtype=np.uint8), depth=np.ones((40, 40)), full_world_xyz=np.ones((40, 40, 3))
    )
    client = Mock(return_value='{"verified":true,"box":[200,200,800,800],"point":[500,500]}')
    _, detections, ids, audit = ground_vlm_region(frame, "lamp", "lamp", client=client, min_depth=0.2, max_depth=4)
    assert ids == [0] and len(detections) == 1
    assert audit["geometry_source"] == "vlm_selected_depth_surface"
    client.return_value = '{"verified":false,"reason":"not visible"}'
    assert ground_vlm_region(frame, "lamp", "lamp", client=client, min_depth=0.2, max_depth=4)[2] == []


def test_invalid_depth_cannot_become_geometry():
    with pytest.raises(ValueError, match="invalid depth"):
        region_depth_mask(np.full((40, 40), np.nan), [0, 0, 1000, 1000], [500, 500], min_depth=0.2, max_depth=4)


def test_geometry_feedback_is_bounded_and_does_not_override_abstention():
    from emet.memory.vlm_region_grounding import select_supported_region

    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    depth = np.ones((40, 40))
    depth[0, 0] = np.nan
    client = Mock(
        side_effect=[
            '{"verified":true,"box":[0,0,1000,1000],"point":[0,0]}',
            '{"verified":true,"box":[0,0,1000,1000],"point":[500,500]}',
        ]
    )
    _, mask, audit = select_supported_region(rgb, depth, "mug", "mug", client=client, min_depth=0.2, max_depth=4)
    assert audit["valid"] and (mask == 0).sum() > 25
    assert client.call_count == 2
    assert len(client.call_args.args[0]) == 3
    assert "invalid depth" in client.call_args.args[0][0]
    client = Mock(return_value='{"verified":false}')
    assert not select_supported_region(rgb, depth, "mug", "mug", client=client, min_depth=0.2, max_depth=4)[2]["valid"]
    client.assert_called_once()


def test_missing_client_is_not_semantic_absence():
    from emet.memory.vlm_region_grounding import select_vlm_region

    with pytest.raises(RuntimeError, match="not initialized"):
        select_vlm_region(np.zeros((8, 8, 3), dtype=np.uint8), "mug", "mug", client=None)


def test_placement_box_prompt_permits_requested_support_and_requires_relation():
    from emet.memory.vlm_region_grounding import select_vlm_region

    client = Mock(return_value='{"verified":false}')
    _, audit = select_vlm_region(
        np.zeros((8, 8, 3), dtype=np.uint8),
        "countertop right of stove",
        "countertop right of stove",
        client=client,
        box_only=True,
        purpose="placement_surface",
    )
    assert "Support furniture" in audit["prompt"]
    assert "wrong side" in audit["prompt"]
    assert "not a hole, occluder or support furniture" not in audit["prompt"]


def test_grounding_prompt_locates_query_only_not_question():
    from emet.memory.vlm_region_grounding import select_vlm_region

    question = "Which of the following is in the bathroom? A) towels B) toilet C) shower D) sink"
    client = Mock(return_value='{"verified":true,"box":[200,200,800,800],"point":[500,500]}')
    _, audit = select_vlm_region(np.zeros((8, 8, 3), dtype=np.uint8), "towels", question, client=client)
    prompt = audit["prompt"]
    assert "Locate the visible object 'towels'" in prompt
    assert question not in prompt
    assert "Verification context" not in prompt
    assert "relationship" not in prompt


def test_box_only_grounding_prompt_locates_query_only_not_question():
    from emet.memory.vlm_region_grounding import select_vlm_region

    question = "Which of the following is in the bathroom? A) towels B) toilet C) shower D) sink"
    client = Mock(return_value='{"verified":true,"box":[200,200,800,800]}')
    _, audit = select_vlm_region(np.zeros((8, 8, 3), dtype=np.uint8), "towels", question, client=client, box_only=True)
    prompt = audit["prompt"]
    assert "Locate the visible object 'towels'" in prompt
    assert question not in prompt
    assert "Verification context" not in prompt
    assert "Do not predict a 3D position" in prompt


def test_grounding_prompt_ignores_description():
    from emet.memory.vlm_region_grounding import select_vlm_region

    client = Mock(return_value='{"verified":true,"box":[200,200,800,800],"point":[500,500]}')
    _, audit = select_vlm_region(np.zeros((8, 8, 3), dtype=np.uint8), "mug", "mug on the bed", client=client)
    assert "Locate the visible object 'mug'" in audit["prompt"]
    assert "on the bed" not in audit["prompt"]


def test_grounding_target_returns_query_only():
    from emet.memory.vlm_region_grounding import _grounding_target

    assert _grounding_target("towels", "Which of the following is in the bathroom? A) towels B) toilet") == "towels"
    assert _grounding_target("mug", "mug") == "mug"
    assert _grounding_target("", "fallback phrase") == "fallback phrase"


def test_shared_vlm_uses_explicit_grounding_prompt_and_budget():
    from unittest.mock import Mock

    from PIL import Image

    from emet.eval.agentic_vlm_assess import _call_eqa_client
    from emet.llms.qwen3_vl_client import Qwen3VLClient

    client = object.__new__(Qwen3VLClient)
    client.generate_multimodal = Mock(return_value='{"verified":false}')
    payload = ["locate", Image.new("RGB", (8, 8))]
    assert _call_eqa_client(client, payload, system_prompt="grounding", max_new_tokens=192)
    client.generate_multimodal.assert_called_once_with(
        payload, system_prompt="grounding", max_new_tokens=192, reset_context=True, assistant_prefill=None
    )


def test_padded_component_search_keeps_original_box_and_requires_semantics():
    from emet.memory.vlm_region_grounding import select_supported_region

    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb[45:55, 45:55] = [200, 30, 30]
    depth = np.ones((100, 100))
    box = [480, 480, 520, 520]  # Too small to supply 25 measured pixels.
    kwargs = {
        "client": Mock(
            side_effect=[
                '{"verified":true,"box":[480,480,520,520]}',
                '{"selected_id":0,"target_unambiguous":true}',
            ]
        ),
        "min_depth": 0.2,
        "max_depth": 4,
        "strategy": "depth_candidates",
        "search_padding": 50,
    }
    parsed, mask, audit = select_supported_region(rgb, depth, "object", "object", **kwargs)
    assert parsed["box"] == box
    assert audit["search_region"] == {"box": [430, 430, 570, 570], "padding": 50}
    assert audit["valid"] and (mask == 0).sum() >= 25
    # No retry/recovery may reinterpret a semantic abstention as success.
    kwargs["client"] = Mock(
        side_effect=[
            '{"verified":true,"box":[480,480,520,520]}',
            '{"selected_id":null,"target_unambiguous":false}',
        ]
    )
    _, mask, audit = select_supported_region(rgb, depth, "object", "object", **kwargs)
    assert not audit["valid"] and (mask == -1).all()
    assert kwargs["client"].call_count == 2


@pytest.mark.parametrize("padding", [-1, 101, True, 5.0])
def test_invalid_search_padding_rejected_before_inference(padding):
    from emet.memory.vlm_region_grounding import select_supported_region

    client = Mock()
    with pytest.raises(ValueError, match="Search padding"):
        select_supported_region(
            np.zeros((40, 40, 3), dtype=np.uint8),
            np.ones((40, 40)),
            "cup",
            "cup",
            client=client,
            min_depth=0.2,
            max_depth=4,
            strategy="depth_candidates",
            search_padding=padding,
        )
    client.assert_not_called()


def test_padded_search_does_not_hide_candidate_overflow():
    from emet.memory.vlm_region_grounding import select_supported_region

    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    depth = np.full((100, 100), np.nan)
    for y in (35, 45, 55):
        for x in (35, 45, 55):
            depth[y : y + 5, x : x + 5] = 1.0
    client = Mock(return_value='{"verified":true,"box":[400,400,600,600]}')
    _, mask, audit = select_supported_region(
        rgb,
        depth,
        "cup",
        "cup",
        client=client,
        min_depth=0.2,
        max_depth=4,
        strategy="depth_candidates",
        search_padding=100,
    )
    assert audit["failure_kind"] == "candidate_overflow"
    assert not audit["valid"] and (mask == -1).all()
    client.assert_called_once()


@pytest.mark.parametrize("box", [[600, 600, 400, 400], [True, 0, 100, 100], [0, 0, 1001, 1000]])
def test_search_padding_cannot_repair_invalid_model_boxes(box):
    import json

    from emet.memory.vlm_region_grounding import select_supported_region

    client = Mock(return_value=json.dumps({"verified": True, "box": box}))
    _, mask, audit = select_supported_region(
        np.zeros((40, 40, 3), dtype=np.uint8),
        np.ones((40, 40)),
        "cup",
        "cup",
        client=client,
        min_depth=0.2,
        max_depth=4,
        strategy="depth_candidates",
        search_padding=50,
    )
    assert audit["failure_kind"] == "invalid_geometry"
    assert (mask == -1).all()
    client.assert_called_once()


def test_search_padding_rejects_incompatible_proposal_source():
    from emet.memory.vlm_region_grounding import select_supported_region

    client = Mock()
    for options in ({"proposal_masks": []}, {"segmenter": Mock()}):
        with pytest.raises(ValueError, match="only supported for RGBD"):
            select_supported_region(
                np.zeros((40, 40, 3), dtype=np.uint8),
                np.ones((40, 40)),
                "cup",
                "cup",
                client=client,
                min_depth=0.2,
                max_depth=4,
                strategy="depth_candidates",
                search_padding=50,
                **options,
            )
    client.assert_not_called()
