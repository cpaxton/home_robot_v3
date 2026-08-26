# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

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
    mem.add_observation(rgb, np.array([4.82, 5.08, 2.46]), ["lamp"], countable_instance=True, identity_key="lamp:1")
    mem.add_observation(rgb, np.array([3.09, 5.02, 1.03]), ["lamp"], countable_instance=True, identity_key="lamp:2")
    mem.add_observation(rgb, np.array([4.68, 2.90, -0.11]), ["table"], countable_instance=True, identity_key="table:1")
    mem.add_observation(rgb, np.array([5.80, 2.91, -0.12]), ["table"], countable_instance=True, identity_key="table:2")
    mem._relevant_objects = ["table lamps"]
    mem._relevant_phrases = ["table lamps"]
    summary = mem._relevant_memory_summary()
    assert "[graph obs 1] at (4.8, 5.1)" in summary
    assert "[graph obs 2] at (3.1, 5.0)" in summary
    assert "lamp at" not in summary.split("nearest:")[0]
    assert "table at" not in summary.split("nearest:")[0]
    assert "4 graph node(s)" not in summary
    assert "list length is not a count" in summary
    assert ": LOOK" in summary
    assert ": PRESENT" not in summary
    mem.record_close_look_label(1, "table lamp")
    looked = mem._relevant_memory_summary()
    assert "table lamp [graph obs 1] at (4.8, 5.1)" in looked
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
    assert "[graph obs 1]" in hint
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


def test_query_answer_noncount_verified_evidence_stays_image_1():
    """A verified non-count submit keeps its evidence as Image 1, not a FIND/look pin.

    For count MCQs the branch intentionally lets FIND pins win over force_obs_ids
    (a single verified frame must not occupy the whole budget), but a plain where-is
    verified answer must not drop its confirmed evidence for a stale Action look or a
    location-FIND pin (regression: force_obs_ids used to be appended last and could be
    displaced entirely at max_images=1).
    """
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    def fake_eqa(_cmds):
        return (
            "reasoning: Image 1 shows the clock above the sink.\n"
            "answer: Above the sink\nconfidence: true\naction:\nconfidence_reasoning: ok"
        )

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=lambda _x: "",
        parameters={"eqa_vl": {"eqa_max_images": 1}},
    )
    mem.add_observation(rgb, np.array([9.0, 9.0, 0.5]), ["chair"], identity_key="hall:1")
    mem.add_observation(rgb, np.array([2.9, -0.4, 1.0]), ["clock"], countable_instance=True)
    mem.add_observation(rgb, np.array([3.0, -0.3, 1.0]), ["clock"], countable_instance=True)
    mem.last_eqa_look_obs_id = 1
    mem._relevant_phrases = ["wall clock"]
    mem._relevant_objects = ["clock"]
    mem._select_relevant_obs_ids = lambda **_k: [1, 2]  # type: ignore[method-assign]
    q = (
        "I'm trying to remember where I placed the wall clock. Where it is in the kitchen? "
        "A) Above the sink B) Next to the refrigerator C) Near the stove "
        "D) On the wall opposite the windows. Answer:"
    )
    _r, _answer, _confidence, _cr, _tp, _imgs = mem.query_answer(q, force_obs_ids=[3])
    assert mem.last_eqa_obs_ids[0] == 3


def _relabel_obs_id(mem: GraphEQAMemory, src: int, dst: int) -> None:
    from dataclasses import replace

    mem._observations = [replace(obs, obs_id=dst) if int(obs.obs_id) == src else obs for obs in mem._observations]
    mem._nodes = [replace(node, obs_id=dst) if int(node.obs_id) == src else node for node in mem._nodes]


def test_count_candidates_match_head_noun_lamp_for_table_lamps():
    """Detector 'lamp' must FIND for 'table lamps' without a close-look name or alias table."""
    mem = GraphEQAMemory(defer_llm_clients=True)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    mem.add_observation(rgb, np.array([1.0, 0.0, 0.5]), ["lamp"], countable_instance=True, identity_key="lamp:1")
    mem.add_observation(rgb, np.array([9.0, 9.0, 0.5]), ["chair"], countable_instance=True, identity_key="chair:1")
    q = "How many table lamps are there? A) One B) Two C) Three D) None. Answer:"
    nodes, target = mem._count_candidate_nodes(q)
    assert target is not None
    assert [int(n.obs_id) for n in nodes] == [1]
    hint = mem._graph_count_hint(q)
    assert "[graph obs 1]" in hint
    assert "GRAPH_COUNT: 1" not in hint


def test_query_answer_count_find_beats_stale_hallway_look():
    """q86: GRAPH_COUNT lamp views must be Image 1, not a leftover Action:55 hallway."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    def fake_eqa(_cmds):
        return "reasoning: r\nanswer: Two\nconfidence: true\naction:\nconfidence_reasoning: ok"

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=lambda _x: "",
        parameters={"eqa_vl": {"eqa_max_images": 1}},
    )
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["chair"], identity_key="hall:1")
    mem.add_observation(
        rgb,
        np.array([4.0, 5.0, 0.5]),
        ["lamp"],
        countable_instance=True,
        identity_key="lamp:1",
    )
    mem.last_eqa_look_obs_id = 1
    mem._select_relevant_obs_ids = lambda **_k: [1]  # type: ignore[method-assign]
    mem.query_answer("How many table lamps are there? A) One B) Two C) Three D) None. Answer:")
    assert mem.last_eqa_obs_ids[0] == 2


def test_query_answer_pins_graph_action_163_as_image_1_among_four_frames():
    """Action:163 with 4 attached display images must be Image 1 on the next call."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    calls = {"n": 0}

    def fake_eqa(_cmds):
        calls["n"] += 1
        if calls["n"] == 1:
            return (
                "reasoning: not in these four frames\n"
                "answer: Unknown\nconfidence: false\naction: 163\n"
                "confidence_reasoning: look at Image 163\n"
            )
        return "reasoning: two lamps\nanswer: Two\nconfidence: true\naction:\nconfidence_reasoning: ok"

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=lambda _x: "",
        parameters={"eqa_vl": {"eqa_max_images": 4}},
    )
    for i, label in enumerate(("chair", "sofa", "wall", "box"), start=1):
        mem.add_observation(
            rgb,
            np.array([float(i), 0.0, 0.5]),
            [label],
            identity_key=f"ctx:{i}",
        )
    mem.add_observation(
        rgb,
        np.array([20.0, 20.0, 0.5]),
        ["box"],
        identity_key="gallery:163",
    )
    _relabel_obs_id(mem, 5, 163)
    mem._select_relevant_obs_ids = lambda **_k: [1, 2, 3, 4]  # type: ignore[method-assign]
    q = "How many table lamps are there? A) One B) Two C) Three D) None. Answer:"
    mem.query_answer(q)
    assert mem.last_eqa_action_obs_id == 163
    assert mem.last_eqa_look_obs_id == 163
    assert 163 not in mem.last_eqa_obs_ids
    mem.query_answer(q, force_obs_ids=[1])
    assert mem.last_eqa_obs_ids[0] == 163


def test_query_answer_count_none_unconfident_when_find_views_unattached():
    """q93: confident None is not final while stool FIND RGB was never attached."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    def fake_eqa(_cmds):
        return (
            "reasoning: stools are not at the kitchen counter in these frames\n"
            "answer: None\nconfidence: true\naction:\n"
            "confidence_reasoning: dining chairs only\n"
        )

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=lambda _x: "",
        parameters={"eqa_vl": {"eqa_max_images": 2}},
    )
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["chair"], identity_key="c1")
    mem.add_observation(rgb, np.array([1.0, 0.0, 0.5]), ["chair"], identity_key="c2")
    mem.add_observation(rgb, np.array([8.0, 8.0, 0.5]), ["stool"], countable_instance=True, identity_key="s1")
    mem.add_observation(rgb, np.array([10.0, 8.0, 0.5]), ["stool"], countable_instance=True, identity_key="s2")
    orig = mem._compose_eqa_answer_obs_ids

    def drop_find(**kwargs):
        kwargs = dict(kwargs)
        kwargs["pin_obs"] = []
        kwargs["look_obs_id"] = None
        return orig(**kwargs)

    mem._compose_eqa_answer_obs_ids = drop_find  # type: ignore[method-assign]
    mem._select_relevant_obs_ids = lambda **_k: [1, 2]  # type: ignore[method-assign]
    q = "How many stools are at the kitchen counter? A) One B) Two C) Three D) None. Answer:"
    _r, answer, confidence, _cr, _tp, _imgs = mem.query_answer(q)
    assert str(answer).strip().lower() == "none"
    assert confidence is False
    assert mem.last_eqa_obs_ids == [1, 2]
    assert mem.last_eqa_look_obs_id in {3, 4}
    assert mem.last_eqa_action_obs_id in {3, 4}


def test_query_answer_none_unconfident_when_other_find_unattached():
    """q12-style: one FIND attached is not enough to finalize None while others remain."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    def fake_eqa(_cmds):
        return (
            "reasoning: no bedside tables in this living room\n"
            "answer: None\nconfidence: true\naction:\n"
            "confidence_reasoning: living room only\n"
        )

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=lambda _x: "",
        parameters={"eqa_vl": {"eqa_max_images": 1}},
    )
    mem.add_observation(
        rgb, np.array([0.0, 0.0, 0.5]), ["sofa"], identity_key="sofa:1"
    )
    mem.add_observation(
        rgb,
        np.array([8.0, 0.0, 0.5]),
        ["nightstand"],
        countable_instance=True,
        identity_key="ns:1",
    )
    mem.add_observation(
        rgb,
        np.array([10.0, 0.0, 0.5]),
        ["nightstand"],
        countable_instance=True,
        identity_key="ns:2",
    )
    orig = mem._compose_eqa_answer_obs_ids

    def only_sofa(**kwargs):
        kwargs = dict(kwargs)
        kwargs["pin_obs"] = []
        kwargs["look_obs_id"] = None
        kwargs["forced"] = [1]
        kwargs["selected"] = [1]
        return orig(**kwargs)

    mem._compose_eqa_answer_obs_ids = only_sofa  # type: ignore[method-assign]
    mem._select_relevant_obs_ids = lambda **_k: [1]  # type: ignore[method-assign]
    q = (
        "How many bedside tables are there in the bedroom with the white bedding? "
        "A) Three B) One C) None D) Two. Answer:"
    )
    _r, answer, confidence, _cr, target, _imgs = mem.query_answer(q)
    assert str(answer).strip().lower() == "none"
    assert confidence is False
    assert mem.last_eqa_obs_ids == [1]
    assert mem.last_eqa_look_obs_id in {2, 3}
    assert target is not None


def test_query_answer_attaches_full_frame_then_labeled_detector_crop():
    """Image 1 is the scene; a leftover slot may add a close-up of that same view."""
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[:, :8] = (180, 20, 20)
    rgb[:, 8:] = (20, 20, 200)
    question = "How many table lamps are there? A) One B) Two C) Three D) None. Answer:"

    captured_one: dict = {}

    def fake_eqa_one(cmds):
        captured_one["cmds"] = cmds
        return "reasoning: hallway with lamp\nanswer: One\nconfidence: true\naction:\nconfidence_reasoning: ok"

    mem_one = GraphEQAMemory(
        eqa_client=fake_eqa_one,
        image_description_client=lambda _x: "lamp",
        parameters={"eqa_vl": {"eqa_max_images": 1}},
    )
    mem_one.add_observation(
        rgb,
        np.array([2.0, 0.0, 0.5]),
        ["lamp"],
        countable_instance=True,
        identity_key="lamp:1",
        bbox_xyxy=(8, 0, 16, 16),
    )
    mem_one.query_answer(question)
    imgs_one = [c for c in captured_one["cmds"] if hasattr(c, "size")]
    assert len(imgs_one) == 1
    scene = np.asarray(imgs_one[0])
    assert scene.shape[1] == 16
    assert scene[:, :, 0].mean() > 40
    assert scene[:, :, 2].mean() > 40

    captured_two: dict = {}

    def fake_eqa_two(cmds):
        captured_two["cmds"] = cmds
        return "reasoning: scene plus close-up\nanswer: One\nconfidence: true\naction:\nconfidence_reasoning: ok"

    mem_two = GraphEQAMemory(
        eqa_client=fake_eqa_two,
        image_description_client=lambda _x: "lamp",
        parameters={"eqa_vl": {"eqa_max_images": 2}},
    )
    mem_two.add_observation(
        rgb,
        np.array([2.0, 0.0, 0.5]),
        ["lamp"],
        countable_instance=True,
        identity_key="lamp:1",
        bbox_xyxy=(8, 0, 16, 16),
    )
    mem_two.query_answer(question)
    imgs_two = [c for c in captured_two["cmds"] if hasattr(c, "size")]
    assert len(imgs_two) == 2
    scene2 = np.asarray(imgs_two[0])
    crop = np.asarray(imgs_two[1])
    assert scene2.shape[1] == 16
    assert crop.shape[1] < 16
    assert crop[:, :, 2].mean() > crop[:, :, 0].mean()
    prompt = " ".join(str(c) for c in captured_two["cmds"] if isinstance(c, str))
    assert "close-up" in prompt
    assert "not another object" in prompt or "second object" in prompt


def test_query_answer_keeps_mixed_recall_when_reserving_closeup_slot():
    """A detector crop steals an extra FIND stool, not the kitchen-counter view."""
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[:, :8] = (180, 20, 20)
    rgb[:, 8:] = (20, 20, 200)
    captured: dict = {}

    def fake_eqa(cmds):
        captured["cmds"] = cmds
        return "reasoning: r\nanswer: Two\nconfidence: true\naction:\nconfidence_reasoning: ok"

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=lambda _x: "stool, kitchen counter",
        parameters={"eqa_vl": {"eqa_max_images": 4}},
    )
    for i in range(4):
        mem.add_observation(
            rgb,
            np.array([float(i), 8.0, 0.5]),
            ["stool"],
            countable_instance=True,
            identity_key=f"stool:{i}",
            bbox_xyxy=(8, 0, 16, 16),
        )
    mem.add_observation(
        rgb,
        np.array([-2.5, 1.0, 0.6]),
        ["kitchen counter"],
        identity_key="counter:1",
    )
    q = "How many stools are at the kitchen counter? A) One B) Two C) Three D) None. Answer:"
    mem.query_answer(q)
    assert 5 in mem.last_eqa_obs_ids
    imgs = [c for c in captured["cmds"] if hasattr(c, "size")]
    assert len(imgs) == len(mem.last_eqa_obs_ids) + 1
    assert np.asarray(imgs[0]).shape[1] == 16
    assert np.asarray(imgs[-1]).shape[1] < 16


def test_query_answer_keeps_relevant_memory_view_when_look_is_hallway():
    """Unconfident time/clock questions still attach the stored clock, not the last nav frame."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    def fake_eqa(_cmds):
        return (
            "reasoning: Image 1 is the wall clock\n"
            "answer: 2-4pm\nconfidence: false\naction:\nconfidence_reasoning: analog"
        )

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=lambda _x: "clock",
        parameters={"eqa_vl": {"eqa_max_images": 2}},
    )
    mem.add_observation(rgb, np.array([-5.0, -2.0, 1.7]), ["clock"], countable_instance=True)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["hallway"])
    mem.last_eqa_look_obs_id = 2
    mem._select_relevant_obs_ids = lambda **_k: [2]  # type: ignore[method-assign]
    q = "What time is it now? A) 8-10am B) 2-4pm C) 5-7pm D) 8-10pm. Answer:"
    mem.query_answer(q)
    assert mem.last_eqa_obs_ids[0] == 1


def test_query_answer_mixes_other_recalled_views_into_find_budget():
    """FIND stools must not occupy every image slot when the question also names a counter."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    def fake_eqa(_cmds):
        return "reasoning: r\nanswer: Two\nconfidence: true\naction:\nconfidence_reasoning: ok"

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=lambda _x: "stool, kitchen counter",
        parameters={"eqa_vl": {"eqa_max_images": 4}},
    )
    for i in range(4):
        mem.add_observation(
            rgb,
            np.array([float(i), 8.0, 0.5]),
            ["stool"],
            countable_instance=True,
            identity_key=f"stool:{i}",
        )
    mem.add_observation(
        rgb,
        np.array([-2.5, 1.0, 0.6]),
        ["kitchen counter"],
        identity_key="counter:1",
    )
    q = "How many stools are at the kitchen counter? A) One B) Two C) Three D) None. Answer:"
    mem.query_answer(q)
    assert 5 in mem.last_eqa_obs_ids


def test_query_answer_does_not_reissue_spent_same_obs_action():
    """Sub-meter revisits of one obs_id are not a new look — pick another target."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    def fake_eqa(_cmds):
        return (
            "reasoning: look at the lamp node\n"
            "answer: Unknown\nconfidence: false\naction: 1\nconfidence_reasoning: hallway"
        )

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=lambda _x: "lamp",
        parameters={"eqa_vl": {"eqa_max_images": 1}},
    )
    mem.add_observation(
        rgb,
        np.array([2.6, -5.9, 2.3]),
        ["lamp"],
        countable_instance=True,
        identity_key="lamp:1",
    )
    q = "How many table lamps are there? A) One B) Two C) Three D) None. Answer:"
    mem.query_answer(q)
    assert mem.last_eqa_action_obs_id == 1
    mem.record_nav_attempt(1, success=True, note="ok_raw", dist_m=3.7)
    mem.record_nav_attempt(1, success=True, note="ok_raw", dist_m=0.25)
    assert mem.eqa_obs_look_spent(1)
    _r, _a, _c, _cr, target, _imgs = mem.query_answer(q)
    assert mem.last_eqa_action_obs_id != 1
    assert target is None
    assert mem.last_eqa_obs_ids[0] == 1


def test_query_answer_visual_find_beats_yolo_lamp_label():
    """q86: SigLIP/grounder view is Image 1 even when YoloE labeled a hallway 'lamp'."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    def fake_eqa(_cmds):
        return "reasoning: two lamps\nanswer: Two\nconfidence: true\naction:\nconfidence_reasoning: ok"

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=lambda _x: "table lamp",
        parameters={"eqa_vl": {"eqa_max_images": 2}},
    )
    mem.add_observation(
        rgb,
        np.array([0.0, 0.0, 0.5]),
        ["lamp"],
        countable_instance=True,
        identity_key="hall-lamp",
    )
    mem.add_observation(rgb, np.array([1.0, 0.0, 0.5]), ["chair"])
    mem.add_observation(
        rgb,
        np.array([4.0, 4.0, 0.5]),
        ["decorative object"],
        identity_key="bed-lamp",
    )
    mem.set_obs_id_grounder(lambda text: 3 if "lamp" in str(text).lower() else None)
    mem.query_answer("How many table lamps are there? A) One B) Two C) Three D) None. Answer:")
    assert mem.last_eqa_obs_ids[0] == 3


def test_eqa_stay_on_attached_clock_view():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([1.0, 1.0, 1.5]), ["clock"])
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["hallway"])
    mem._relevant_objects = ["clock"]
    mem._relevant_phrases = ["clock"]
    mem.last_eqa_obs_ids = [1]
    assert mem.eqa_stay_on_attached_view() is True
    mem.last_eqa_obs_ids = [2]
    assert mem.eqa_stay_on_attached_view() is False


def test_eqa_stay_read_action_uses_later_image_not_only_image1():
    """VLM ``read 2`` stays on that attached view even when Image 1 is a landmark."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["refrigerator"])
    mem.add_observation(rgb, np.array([2.0, 1.0, 1.5]), ["clock"])
    mem._question = "I'm trying to remember where I placed the wall clock. Where it is in the kitchen?"
    mem.last_eqa_obs_ids = [1, 2]
    mem.last_eqa_parsed = ("", "Unknown", False, "read 2", "hands too small")
    assert mem.eqa_stay_on_attached_view() is True
    assert mem.eqa_attached_target_obs_id() == 2
    mem.last_eqa_parsed = ("", "Unknown", False, "", "")
    mem.last_eqa_obs_ids = [1]
    assert mem.eqa_stay_on_attached_view() is False


def test_eqa_stay_read_action_works_for_sign_or_oven():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["door"])
    mem.add_observation(rgb, np.array([2.0, 1.0, 1.5]), ["sign"])
    mem.add_observation(rgb, np.array([4.0, 1.0, 1.0]), ["oven"])
    mem.last_eqa_obs_ids = [1, 2]
    mem.last_eqa_parsed = ("", "Unknown", False, "read 2", "letters too small")
    assert mem.eqa_attached_target_obs_id() == 2
    mem.last_eqa_obs_ids = [1, 3]
    mem.last_eqa_parsed = ("", "Unknown", False, "read 2", "digits too small")
    assert mem.eqa_attached_target_obs_id() == 3


def test_visual_find_fn_topk_spreads_same_noun_xy():
    """Argmax FIND would keep two railing stools; top-k + XY spread includes the island."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    def fake_eqa(_cmds):
        return "reasoning: stools\nanswer: Two\nconfidence: true\naction:\nconfidence_reasoning: ok"

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=lambda _x: "stool",
        parameters={"eqa_vl": {"eqa_max_images": 2}},
    )
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["railing"])
    mem.add_observation(rgb, np.array([0.2, 0.0, 0.5]), ["railing"])
    mem.add_observation(rgb, np.array([8.0, 8.0, 0.5]), ["island"])
    mem.set_visual_find_fn(lambda phrase, max_n: [(0.40, 1), (0.38, 2), (0.33, 3)][:max_n])
    mem.query_answer("How many stools are there? A) One B) Two C) Three D) Four. Answer:")
    assert mem.last_eqa_obs_ids[0] == 1
    assert 3 in mem.last_eqa_obs_ids


def test_eqa_stay_uses_visual_find_not_yolo_hallway_lamp():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["lamp"])
    mem.add_observation(rgb, np.array([4.0, 4.0, 0.5]), ["decorative object"])
    mem._relevant_objects = ["table lamp"]
    mem._relevant_phrases = ["table lamp"]
    mem.set_visual_find_fn(lambda phrase, max_n: [(0.30, 2)])
    mem.last_eqa_obs_ids = [1]
    assert mem.eqa_stay_on_attached_view() is False
    mem.last_eqa_obs_ids = [2]
    assert mem.eqa_stay_on_attached_view() is True


def test_eqa_approach_attached_find_when_far_then_stay_when_close():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([8.0, 8.0, 1.5]), ["clock"])
    mem._relevant_objects = ["clock"]
    mem._relevant_phrases = ["clock"]
    mem.last_eqa_obs_ids = [1]
    far = mem.eqa_approach_attached_find(np.array([0.0, 0.0, 0.0]))
    assert far is not None
    assert float(np.linalg.norm(np.asarray(far[:2]) - np.array([0.0, 0.0]))) > 1.0
    close = mem.eqa_approach_attached_find(np.array([8.0, 8.0, 0.0]))
    assert close is None
    mem.last_eqa_parsed = ("", "Unknown", False, "read 1", "too small")
    # Zoom is off by default, so read N does not stay for a crop.
    assert mem.eqa_approach_attached_find(np.array([0.0, 0.0, 0.0])) is not None


def test_eqa_approach_stays_on_read_when_center_zoom_enabled(monkeypatch):
    monkeypatch.setenv("EMET_EQA_CENTER_ZOOM", "1")
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([8.0, 8.0, 1.5]), ["clock"])
    mem._relevant_objects = ["clock"]
    mem._relevant_phrases = ["clock"]
    mem.last_eqa_obs_ids = [1]
    mem.last_eqa_parsed = ("", "Unknown", False, "read 1", "too small")
    assert mem.eqa_approach_attached_find(np.array([0.0, 0.0, 0.0])) is None


def test_format_eqa_view_status_exposes_visit_counters():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["sofa"])
    mem.add_observation(rgb, np.array([4.0, 4.0, 0.5]), ["table"])
    mem.last_eqa_obs_ids = [1, 2]
    mem._history_outputs = [
        "Iter: answer=Unknown conf=false action=1 salvage=0 | living room",
        "Iter: answer=Unknown conf=false action=1 salvage=0 | still no table",
    ]
    mem._obs_nav_dists[1] = [2.0, 0.5, 0.2]
    block = mem.format_eqa_view_status([1, 2])
    assert "VIEW_STATUS" in block
    assert "Image 1 (obs 1): visits=3" in block
    assert "look=2" in block
    assert "unknown=2" in block


def test_format_eqa_view_status_counts_read_actions():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["sofa"])
    mem.add_observation(rgb, np.array([4.0, 4.0, 0.5]), ["table"])
    mem.last_eqa_obs_ids = [1, 2]
    mem._history_outputs = [
        mem.format_eqa_history_outcome(
            answer="Unknown", confidence=False, action="read 2", reasoning="letters too small"
        ),
        mem.format_eqa_history_outcome(
            answer="Unknown", confidence=False, action="read 2", reasoning="still too small"
        ),
    ]
    block = mem.format_eqa_view_status([1, 2])
    assert "Image 2 (obs 2):" in block
    assert "read=2" in block
    assert "look=0" in block


def test_eqa_should_stay_releases_on_unknown_find_not_read():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["chair"])
    mem._relevant_objects = ["dining chair"]
    mem._relevant_phrases = ["dining chair"]
    mem.last_eqa_obs_ids = [1]
    mem.last_eqa_parsed = ("", "Unknown", False, "1", "no table visible")
    assert mem.eqa_stay_on_attached_view() is True
    assert mem.eqa_should_stay_on_attached_view(answer="Unknown", confidence=False) is False


def test_eqa_should_stay_allows_read_unknown_until_spent(monkeypatch):
    monkeypatch.setenv("EMET_EQA_CENTER_ZOOM", "1")
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["sign"])
    mem.add_observation(rgb, np.array([2.0, 1.0, 1.5]), ["sign"])
    mem.last_eqa_obs_ids = [1, 2]
    mem.last_eqa_parsed = ("", "Unknown", False, "read 2", "too small")
    assert mem.eqa_should_stay_on_attached_view(answer="Unknown", confidence=False) is True


def test_eqa_should_stay_releases_read_when_zoom_disabled():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["sign"])
    mem.last_eqa_obs_ids = [1]
    mem.last_eqa_parsed = ("", "Unknown", False, "read 1", "too small")
    assert mem.eqa_should_stay_on_attached_view(answer="Unknown", confidence=False) is False


def test_graph_count_hint_prefers_visual_find_images():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(
        rgb,
        np.array([0.0, 0.0, 0.5]),
        ["lamp"],
        countable_instance=True,
        identity_key="hall-lamp",
    )
    mem.add_observation(rgb, np.array([4.0, 4.0, 0.5]), ["decorative object"])
    mem._relevant_objects = ["table lamp"]
    mem._relevant_phrases = ["table lamp"]
    mem.set_visual_find_fn(lambda phrase, max_n: [(0.40, 2)])
    q = "How many table lamps are there? A) One B) Two C) Three D) None. Answer:"
    hint = mem._graph_count_hint(q)
    assert "obs2" in hint
    assert "GRAPH_COUNT: 1" not in hint


def test_query_answer_highlight_adds_clock_when_time_phrase_misses():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    def client(cmds):
        if isinstance(cmds, list) and any(isinstance(c, Image.Image) for c in cmds):
            return "wall clock"
        return "time"

    def fake_eqa(_cmds):
        return "reasoning: 2-4pm\nanswer: 2-4pm\nconfidence: true\naction:\nconfidence_reasoning: ok"

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=client,
        parameters={"eqa_vl": {"eqa_max_images": 2}},
    )
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["hallway"])
    mem.add_observation(rgb, np.array([3.0, 1.0, 1.5]), ["decorative object"])
    mem.set_obs_id_grounder(lambda text: 2 if "clock" in str(text).lower() else None)
    mem.query_answer("What time is it now? A) 1-3pm B) 2-4pm C) 5-7pm D) Unknown. Answer:")
    assert mem.last_eqa_obs_ids[0] == 2
    assert any("clock" in str(p).lower() for p in (mem._relevant_objects or []))


def test_query_answer_skips_highlight_when_visual_find_already_hits():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    image_calls = {"n": 0}

    def client(cmds):
        if isinstance(cmds, list) and any(isinstance(c, Image.Image) for c in cmds):
            image_calls["n"] += 1
            return "wrong object"
        return "stool"

    def fake_eqa(_cmds):
        return "reasoning: two\nanswer: Two\nconfidence: true\naction:\nconfidence_reasoning: ok"

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=client,
        parameters={"eqa_vl": {"eqa_max_images": 2}},
    )
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["railing"])
    mem.add_observation(rgb, np.array([8.0, 8.0, 0.5]), ["island"])
    mem.set_visual_find_fn(lambda phrase, max_n: [(0.40, 1), (0.30, 2)][:max_n])
    mem.query_answer("How many stools are there? A) One B) Two C) Three D) Four. Answer:")
    assert image_calls["n"] == 0
    assert mem.last_eqa_obs_ids[0] == 1


def test_query_answer_location_mcq_keeps_landmark_over_visual_find():
    """Where-is trash: dining-table SigLIP must not steal Image 1 from the fridge landmark."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    def fake_eqa(_cmds):
        return (
            "reasoning: Image 1 is the refrigerator.\n"
            "answer: Next to the refrigerator\nconfidence: true\naction:\nconfidence_reasoning: ok"
        )

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=lambda _x: "trash can",
        parameters={"eqa_vl": {"eqa_max_images": 2}},
    )
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["sofa"])
    mem.add_observation(rgb, np.array([2.0, 1.0, 0.5]), ["refrigerator"])
    mem.set_visual_find_fn(lambda phrase, max_n: [(0.50, 1)])
    mem._select_relevant_obs_ids = lambda **_k: [2]  # type: ignore[method-assign]
    mem.query_answer(
        "Where is the trash can? A) Next to the dining table B) Next to the TV "
        "C) Next to the kitchen sink D) Next to the refrigerator. Answer:"
    )
    assert mem.last_eqa_obs_ids[0] == 2


def test_resolve_voxel_frame_to_graph_obs_id_matches_camera():
    rgb_a = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb_b = np.ones((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(
        rgb_a,
        np.array([1.0, 0.0, 0.5]),
        ["chair"],
        viewer_xyz=np.array([0.0, 0.0, 1.0]),
    )
    mem.add_observation(
        rgb_b,
        np.array([9.0, 9.0, 0.5]),
        ["stool"],
        viewer_xyz=np.array([4.0, 4.0, 1.0]),
    )

    def _frame(rgb, cam):
        pose = np.eye(4)
        pose[:3, 3] = cam
        return SimpleNamespace(rgb=rgb, camera_pose=pose)

    vm = SimpleNamespace(
        observations=[
            _frame(rgb_a, [0.0, 0.0, 1.0]),
            _frame(rgb_b, [4.0, 4.0, 1.0]),
        ]
    )
    assert mem.resolve_voxel_frame_to_graph_obs_id(1, vm) == 1
    assert mem.resolve_voxel_frame_to_graph_obs_id(2, vm) == 2
    assert mem.nearest_graph_obs_to_xyz([9.0, 9.0, 0.5]) == 2


def test_visual_find_ranks_survive_encoder_release():
    """Habitat drops SigLIP before query_answer; FIND must use the warmed cache."""
    from emet.eval.dynagraph_vram import prepare_dynagraph_vram_for_eqa

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.memory_summary_enabled = True
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["railing"])
    mem.add_observation(rgb, np.array([0.2, 0.0, 0.5]), ["railing"])
    mem.add_observation(rgb, np.array([8.0, 8.0, 0.5]), ["island"])
    mem._relevant_objects = ["stool"]
    mem._relevant_phrases = ["kitchen island stools"]
    calls = {"n": 0}

    def _find(phrase, max_n=4):
        calls["n"] += 1
        return [(0.40, 1), (0.38, 2), (0.33, 3)][:max_n]

    mem.set_visual_find_fn(_find)

    class _Enc:
        def encode_image(self, _rgb):
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

        def encode_text(self, _text):
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    enc = _Enc()
    agent = SimpleNamespace(
        encoder=enc,
        voxel_map=SimpleNamespace(encoder=enc),
        graph_memory=mem,
        _eqa_question="How many stools are there at the kitchen island?",
    )
    prepare_dynagraph_vram_for_eqa(agent)
    assert agent.encoder is None
    assert agent.voxel_map.encoder is None
    assert calls["n"] >= 1
    assert "stool" in mem._visual_find_rank_cache

    def _dead(*_a, **_k):
        raise RuntimeError("GPU SigLIP released")

    mem.set_visual_find_fn(_dead)
    ids = mem._visual_find_obs_ids(["stool"], max_n=2)
    assert ids[0] == 1
    assert 3 in ids
