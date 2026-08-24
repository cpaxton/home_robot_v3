# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from emet.memory.graph_eqa.graph_memory import GraphEQAMemory, countable_primary_label_matches
from emet.memory.graph_eqa.graph_object_fusion.config import GraphObjectFusionConfig
from emet.memory.graph_eqa.graph_object_fusion.fusion import GraphDetectionCandidate, GraphObjectFusion
from emet.memory.graph_eqa.instance_observations import (
    filter_detections_for_graph_admission,
    frame_instances_to_detections,
)


class _MockYoloEVocab:
    class_list = [f"cls_{i}" for i in range(200)]


def test_frame_instances_propagate_score_and_support():
    h, w = 8, 8
    inst = torch.full((h, w), -1, dtype=torch.long)
    inst[0:4, 0:4] = 0
    fw = torch.zeros(h, w, 3, dtype=torch.float32)
    fw[0:4, 0:4] = torch.tensor([1.0, 2.0, 3.0])
    depth = torch.ones(h, w) * 0.5
    classes = torch.tensor([2], dtype=torch.long)
    scores = torch.tensor([0.42], dtype=torch.float32)
    frame = SimpleNamespace(
        instance=inst,
        full_world_xyz=fw,
        depth=depth,
        instance_classes=classes,
        instance_scores=scores,
    )
    out = frame_instances_to_detections(
        frame,
        min_depth=0.01,
        max_depth=10.0,
        detection_model=_MockYoloEVocab(),
        min_points=4,
    )
    assert len(out) == 1
    assert abs(out[0]["detection_score"] - 0.42) < 1e-5
    assert out[0]["mask_point_count"] == 16


def test_filter_detections_for_graph_admission():
    cfg = GraphObjectFusionConfig(instance_min_confidence=0.2, instance_min_mask_points=10)
    dets = [
        {"detection_score": 0.1, "mask_point_count": 100},
        {"detection_score": 0.5, "mask_point_count": 5},
        {"detection_score": 0.5, "mask_point_count": 50},
    ]
    kept, stats = filter_detections_for_graph_admission(dets, config=cfg)
    assert len(kept) == 1
    assert stats["rejected_confidence"] == 1
    assert stats["rejected_support"] == 1
    assert stats["admitted"] == 1


def test_fusion_keeps_incompatible_instance_labels_distinct():
    cfg = GraphObjectFusionConfig(
        enabled=True,
        spatial_merge_xy_m=0.55,
        fallback_spatial_merge_xy_m=0.55,
        require_label_match=False,
        require_label_match_for_instances=True,
    )
    fusion = GraphObjectFusion(cfg)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.spatial_merge_m = 0.0
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    fusion.apply_detection(
        mem,
        rgb,
        GraphDetectionCandidate(
            label="bathroom stall",
            xyz=np.array([1.0, 0.0, 0.5]),
            countable_instance=True,
        ),
    )
    fusion.apply_detection(
        mem,
        rgb,
        GraphDetectionCandidate(
            label="table lamp",
            xyz=np.array([1.05, 0.02, 0.52]),
            countable_instance=True,
        ),
    )
    objs = [n for n in mem.get_nodes() if not n.is_viewpoint and not n.is_frontier]
    assert len(objs) == 2


def test_semantic_hypothesis_does_not_merge_into_countable_node():
    cfg = GraphObjectFusionConfig(
        enabled=True,
        spatial_merge_xy_m=0.55,
        fallback_spatial_merge_xy_m=0.55,
        require_label_match=False,
        require_label_match_for_instances=True,
    )
    fusion = GraphObjectFusion(cfg)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.spatial_merge_m = 0.0
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    fusion.apply_detection(
        mem,
        rgb,
        GraphDetectionCandidate(label="table lamp", xyz=np.array([1.0, 0.0, 0.5]), countable_instance=True),
    )
    fusion.apply_detection(
        mem,
        rgb,
        GraphDetectionCandidate(label="bed", xyz=np.array([1.02, 0.01, 0.51]), semantic_only=True),
    )
    objs = [n for n in mem.get_nodes() if not n.is_viewpoint and not n.is_frontier]
    assert len(objs) == 2
    lamp = next(n for n in objs if n.labels == ["table lamp"])
    assert lamp.countable_instance is True
    assert not any("bed" in lab for lab in lamp.labels)


def test_countable_primary_label_matches_ignores_secondary_labels():
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.array([1.0, 0.0, 0.5]),
        ["bathroom stall", "table lamp"],
        countable_instance=True,
    )
    node = [n for n in mem.get_nodes() if not n.is_viewpoint][0]
    assert countable_primary_label_matches("table lamp", node) is False
    assert countable_primary_label_matches("bathroom stall", node) is True


def test_confirmed_memory_lists_views_without_detector_class_names():
    """q86: do not advertise YoloE 'lamp'/'table' strings or a memory integer."""
    mem = GraphEQAMemory(defer_llm_clients=True)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    mem.add_observation(
        rgb, np.array([4.82, 5.08, 2.46]), ["lamp"], countable_instance=True, identity_key="lamp:1"
    )
    mem.add_observation(
        rgb, np.array([3.09, 5.02, 1.03]), ["lamp"], countable_instance=True, identity_key="lamp:2"
    )
    mem.add_observation(
        rgb, np.array([4.68, 2.90, -0.11]), ["table"], countable_instance=True, identity_key="table:1"
    )
    mem.add_observation(
        rgb, np.array([5.80, 2.91, -0.12]), ["table"], countable_instance=True, identity_key="table:2"
    )
    mem._relevant_objects = ["table lamps"]
    mem._relevant_phrases = ["table lamps"]
    summary = mem._relevant_memory_summary()
    assert "[Image 1] at (4.8, 5.1)" in summary
    assert "[Image 2] at (3.1, 5.0)" in summary
    assert "lamp at" not in summary.split("nearest:")[0]
    assert "table at" not in summary.split("nearest:")[0]
    assert "4 graph node(s)" not in summary
    assert "list length is not a count" in summary
    assert ": LOOK" in summary
    assert ": PRESENT" not in summary
    mem.record_close_look_label(1, "table lamp")
    looked = mem._relevant_memory_summary()
    assert "table lamp [Image 1] at (4.8, 5.1)" in looked
    q = "How many table lamps are there in the bedroom? A) Three B) Four C) One D) Two. Answer:"
    hint = mem._graph_count_hint(q)
    assert "GRAPH_COUNT: 4" not in hint
    if hint:
        assert "not an exact count" in hint
        assert "do not use this list length as the answer" in hint


def test_scene_graph_uses_close_look_label():
    mem = GraphEQAMemory(defer_llm_clients=True)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    mem.add_observation(rgb, np.array([1.0, 0.0, 0.5]), ["lamp"], countable_instance=True)
    mem.record_close_look_label(1, "table lamp")
    graph = mem.to_string()
    assert "table lamp" in graph
    from emet.memory.graph_eqa.pretty_print import format_scene_graph_pretty

    pretty = format_scene_graph_pretty(mem)
    assert "table lamp" in pretty
    from emet.visualization.rerun import graph_node_primary_label

    obj = next(n for n in mem.get_nodes() if not n.is_viewpoint and not n.is_frontier)
    assert graph_node_primary_label(obj) == "table lamp"


def test_close_look_label_tags_seen_from_object():
    """Fusion may keep an earlier obs_id; close look still stamps the seen object."""
    from emet.memory.graph_eqa.graph_memory import GraphNode

    mem = GraphEQAMemory(defer_llm_clients=True)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    mem.add_observation(
        rgb,
        np.array([1.0, 0.0, 0.5]),
        ["lamp"],
        countable_instance=True,
        viewer_xyz=np.array([0.0, 0.0, 0.0]),
    )
    obj = next(n for n in mem.get_nodes() if not n.is_viewpoint and not n.is_frontier)
    vp = GraphNode(
        node_id=max(int(n.node_id) for n in mem.get_nodes()) + 1,
        labels=["view img 9"],
        xyz=np.array([0.2, 0.0, 0.0]),
        obs_id=9,
        is_viewpoint=True,
    )
    mem._nodes.append(vp)
    mem._edges.append((int(obj.node_id), int(vp.node_id), "seen_from"))
    mem.record_close_look_label(9, "bedside lamp")
    stamped = next(n for n in mem.get_nodes() if int(n.node_id) == int(obj.node_id))
    assert stamped.close_look_label == "bedside lamp"


def test_query_answer_pins_count_candidate_views():
    """Count MCQs attach FIND candidate RGB even if diversification picked something else."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    captured: dict = {}

    def fake_eqa(cmds):
        captured["cmds"] = cmds
        return "reasoning: r\nanswer: Two\nconfidence: true\naction:\nconfidence_reasoning: ok"

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=lambda _x: "",
        parameters={"eqa_vl": {"eqa_max_images": 1}},
    )
    mem.add_observation(
        rgb,
        np.array([0.0, 0.0, 0.5]),
        ["umbrella"],
        identity_key="u1",
        countable_instance=True,
    )
    mem.add_observation(rgb, np.array([9.0, 9.0, 0.5]), ["chair"])
    mem._select_relevant_obs_ids = lambda **_k: [2]  # type: ignore[method-assign]
    mem.query_answer("How many umbrellas are there? A) One B) Two C) Three D) Four. Answer:")
    assert mem.last_eqa_obs_ids[0] == 1
    assert any(isinstance(c, str) and "GRAPH_COUNT:" in c and "views to look at" in c for c in captured["cmds"])


def test_query_answer_count_pins_survive_forced_single_image():
    """Agentic submit max_images=1 must not drop FIND lamp/stool RGB (q86 bathroom)."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    def fake_eqa(_cmds):
        return "reasoning: r\nanswer: Two\nconfidence: true\naction:\nconfidence_reasoning: ok"

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=lambda _x: "",
        parameters={"eqa_vl": {"eqa_max_images": 1}},
    )
    mem.add_observation(
        rgb,
        np.array([0.0, 0.0, 0.5]),
        ["umbrella"],
        identity_key="u1",
        countable_instance=True,
    )
    mem.add_observation(rgb, np.array([9.0, 9.0, 0.5]), ["chair"])
    mem._select_relevant_obs_ids = lambda **_k: [2]  # type: ignore[method-assign]
    mem.query_answer(
        "How many umbrellas are there? A) One B) Two C) Three D) Four. Answer:",
        force_obs_ids=[2],
    )
    assert mem.last_eqa_obs_ids[0] == 1


def test_query_answer_pins_previous_action_obs_as_image_1():
    """Action: graph obs_id must be Image 1 on the next query_answer (q86 Image 163)."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    calls = {"n": 0}

    def fake_eqa(_cmds):
        calls["n"] += 1
        if calls["n"] == 1:
            return (
                "reasoning: lamp not in these frames\n"
                "answer: One\nconfidence: false\naction: 3\n"
                "confidence_reasoning: look at Image 3\n"
            )
        return "reasoning: two lamps\nanswer: Two\nconfidence: true\naction:\nconfidence_reasoning: ok"

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=lambda _x: "",
        parameters={"eqa_vl": {"eqa_max_images": 1}},
    )
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["chair"])
    mem.add_observation(rgb, np.array([1.0, 0.0, 0.5]), ["sofa"])
    mem.add_observation(
        rgb,
        np.array([2.0, 0.0, 0.5]),
        ["lamp"],
        identity_key="lamp:1",
        countable_instance=True,
    )
    mem._select_relevant_obs_ids = lambda **_k: [1]  # type: ignore[method-assign]
    q = "How many table lamps are there? A) One B) Two C) Three D) None. Answer:"
    mem.query_answer(q)
    assert mem.last_eqa_action_obs_id == 3
    mem.query_answer(q, force_obs_ids=[1])
    assert mem.last_eqa_obs_ids[0] == 3


def test_count_hint_matches_close_look_name_not_just_detector():
    """YoloE 'lamp' + Qwen 'table lamp' must still FIND the view for a table-lamp count."""
    mem = GraphEQAMemory(defer_llm_clients=True)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    mem.add_observation(
        rgb,
        np.array([1.0, 0.0, 0.5]),
        ["lamp"],
        countable_instance=True,
        identity_key="lamp:1",
    )
    mem.add_observation(
        rgb,
        np.array([9.0, 9.0, 0.5]),
        ["chair"],
        countable_instance=True,
        identity_key="chair:1",
    )
    mem.record_close_look_label(1, "table lamp")
    q = "How many table lamps are there in the bedroom? A) Three B) Four C) One D) Two. Answer:"
    nodes, target = mem._count_candidate_nodes(q)
    assert target is not None
    assert [int(n.obs_id) for n in nodes] == [1]
    hint = mem._graph_count_hint(q)
    assert "[Image 1]" in hint
    assert "GRAPH_COUNT: 1" not in hint


def test_query_answer_pins_location_object_view_ahead_of_landmark():
    """q47-style: fridge landmark must not steal Image 1 from the clock view."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    captured: dict = {}

    def fake_eqa(cmds):
        captured["cmds"] = cmds
        return (
            "reasoning: Image 1 shows the clock above the sink.\n"
            "answer: Above the sink\nconfidence: true\naction:\nconfidence_reasoning: ok"
        )

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=lambda _x: "",
        parameters={"eqa_vl": {"eqa_max_images": 2}},
    )
    mem.memory_summary_enabled = True
    mem.add_observation(rgb, np.array([2.9, -0.4, 1.0]), ["clock"], countable_instance=True)
    mem.add_observation(rgb, np.array([3.1, -0.2, 1.0]), ["refrigerator"])
    mem._relevant_phrases = ["wall clock"]
    mem._relevant_objects = ["clock"]
    mem._select_relevant_obs_ids = lambda **_k: [2]  # type: ignore[method-assign]
    q = (
        "I'm trying to remember where I placed the wall clock. Where it is in the kitchen? "
        "A) Above the sink B) Next to the refrigerator C) Near the stove "
        "D) On the wall opposite the windows. Answer:"
    )
    mem.query_answer(q)
    assert mem.last_eqa_obs_ids[0] == 1
