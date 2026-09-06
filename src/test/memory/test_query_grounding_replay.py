# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.memory.query_grounding import cache_grounding_record, replay_grounding_admission, select_query_detections


def detection():
    return {
        "instance_id": 0,
        "label_short": "lamp",
        "bbox_xyxy": [0, 0, 7, 7],
        "detection_score": 0.6,
        "mask_point_count": 64,
    }


def test_relation_requires_visual_verification_not_shared_words():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    assert select_query_detections("lamp bed", "lamp on the bed", [detection()], rgb)[0] == []
    client = Mock(return_value='{"matching_ids": [0], "constraints_verified": true}')
    assert select_query_detections("lamp bed", "lamp on the bed", [detection()], rgb, client)[0] == [0]
    assert len(client.call_args.args[0]) == 3  # prompt, original, numbered regions


@pytest.mark.parametrize(
    "reply",
    [
        "{}",
        '{"matching_ids": [99], "constraints_verified": true}',
        '{"matching_ids": [0], "constraints_verified": "true"}',
        '{"matching_ids": [0, 0], "constraints_verified": true}',
    ],
)
def test_semantic_selection_abstains_on_invalid_output(reply):
    ids, _ = select_query_detections(
        "lamp bed", None, [detection()], np.zeros((8, 8, 3), dtype=np.uint8), Mock(return_value=reply)
    )
    assert ids == []


def test_cache_replays_admission_without_models(tmp_path):
    from emet.memory.graph_eqa.graph_object_fusion.config import GraphObjectFusionConfig

    cache_grounding_record(
        tmp_path,
        query="lamp",
        revision=2,
        source_obs_id=1,
        detections=[detection()],
        matching_ids=[0],
        verification={"source": "exact_label"},
    )
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    cfg = GraphObjectFusionConfig(enabled=True, instance_min_confidence=0.5, instance_min_mask_points=10)
    assert replay_grounding_admission(record, cfg)
    cfg.instance_min_confidence = 0.7
    assert not replay_grounding_admission(record, cfg)


def test_voxel_router_client_does_not_attach_graph():
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = SimpleNamespace(parameters={}, graph_memory=None)
    ex = AgenticEQAExecutor(agent, "Where is the lamp?", router=False)
    client = Mock()
    ex._voxel_eqa_client = client
    assert ex.eqa_client is client
    assert agent.graph_memory is None


def test_stalled_inspection_uses_existing_fallback_without_model_call():
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    ex = AgenticEQAExecutor(SimpleNamespace(parameters={}, graph_memory=None), "Where is the lamp?", router=False)
    ex._unchanged_inspections = 2
    ex._fallback_tool = Mock(return_value=("explore_frontier", {}))
    calls, source, _ = ex._route_tool_calls()
    assert calls == [("explore_frontier", {})]
    assert source == "inspection_no_progress"
