# Copyright (c) Chris Paxton 2026

from types import SimpleNamespace

import numpy as np

from emet.memory.graph_eqa.graph_memory import GraphEQAMemory


def _rgb():
    return np.zeros((8, 8, 3), dtype=np.uint8)


def test_context_relations_have_timestamped_beliefs():
    memory = GraphEQAMemory(defer_llm_clients=True)
    room_obs = memory.add_observation(_rgb(), np.array([0.0, 0.0, 1.0]), ["kitchen"])
    room = memory._node_for_obs(room_obs)
    assert room is not None
    room.bounds_3d = {
        "min": [-3.0, -3.0, 0.0],
        "max": [3.0, 3.0, 3.0],
        "center": [0.0, 0.0, 1.5],
        "size": [6.0, 6.0, 3.0],
    }
    basket_obs = memory.add_observation(
        _rgb(),
        np.array([1.0, 1.0, 0.5]),
        ["woven basket"],
        viewer_xyz=np.array([1.5, 1.5, 0.0]),
    )
    memory._update_edges()
    basket = memory._node_for_obs(basket_obs)
    assert basket is not None
    assert (room.node_id, basket.node_id, "contains") in memory.get_edges()
    relation = memory._relation_beliefs[(room.node_id, basket.node_id, "contains")]
    assert relation.confidence >= 0.8
    assert relation.last_evidence_step >= 0
    assert any(edge[2] == "accessible_from" for edge in memory.get_edges())


def test_information_gain_ranking_reports_components():
    memory = GraphEQAMemory(defer_llm_clients=True)
    memory._relevant_phrases = ["basket"]
    first = memory.add_observation(_rgb(), np.array([1.0, 0.0, 0.5]), ["basket", "dining table"])
    second = memory.add_observation(_rgb(), np.array([0.2, 0.0, 0.5]), ["basket"])
    second_node = memory._node_for_obs(second)
    assert second_node is not None
    second_node.nav_attempts = 2
    second_node.nav_failures = 2
    ranked = memory.hypothesize_nav_targets(
        "Where is the basket? A) by dining table B) by sofa Answer:",
        max_k=2,
        robot_xyt=np.array([0.0, 0.0, 0.0]),
    )
    assert ranked[0].obs_id == first
    assert ranked[0].answerability_gain >= ranked[1].answerability_gain
    assert ranked[1].failure_risk == 1.0
    assert ranked[0].path_cost > 0.0


def test_position_contradiction_creates_change_event_without_midpoint():
    memory = GraphEQAMemory(defer_llm_clients=True)
    obs_id = memory.add_observation(_rgb(), np.array([0.0, 0.0, 0.5]), ["mug"])
    node = memory._node_for_obs(obs_id)
    assert node is not None
    candidate = SimpleNamespace(
        label="mug",
        xyz=np.array([2.0, 0.0, 0.5]),
        bbox_xyxy=None,
        bounds_3d=None,
        embedding=None,
    )
    memory.merge_object_detection(
        _rgb(),
        candidate,
        merge_into_node_id=node.node_id,
    )
    updated = memory._node_for_obs(obs_id)
    assert updated is not None
    assert np.allclose(updated.xyz, candidate.xyz)
    assert updated.change_events[-1]["type"] == "position_contradiction"
    assert len(updated.position_history) >= 2


def test_revisited_view_detects_missing_object_without_oracle():
    memory = GraphEQAMemory(defer_llm_clients=True)
    viewer = np.array([0.0, 0.0, 0.0])
    obs_id = memory.add_observation(
        _rgb(),
        np.array([1.0, 0.0, 0.5]),
        ["mug"],
        viewer_xyz=viewer,
    )
    assert memory.observe_visible_labels([], viewer, step=2) == []
    events = memory.observe_visible_labels([], viewer, step=3)
    assert len(events) == 1
    assert events[0]["type"] == "expected_object_missing"
    node = memory._node_for_obs(obs_id)
    assert node is not None
    assert node.expected_absence_count == 2
    assert memory.get_change_events()[-1]["node_id"] == node.node_id


def test_dynamic_beliefs_roundtrip_checkpoint(tmp_path):
    from emet.memory.adapters import GraphEQABackend

    memory = GraphEQAMemory(defer_llm_clients=True)
    obs_id = memory.add_observation(
        _rgb(),
        np.array([1.0, 0.0, 0.5]),
        ["mug"],
        viewer_xyz=np.array([0.0, 0.0, 0.0]),
    )
    memory.observe_visible_labels([], np.array([0.0, 0.0, 0.0]), step=2)
    memory.observe_visible_labels([], np.array([0.0, 0.0, 0.0]), step=3)
    GraphEQABackend(memory).save(str(tmp_path), final_step=3)

    loaded = GraphEQAMemory(defer_llm_clients=True)
    GraphEQABackend(loaded).load(str(tmp_path))
    node = loaded._node_for_obs(obs_id)
    assert node is not None
    assert node.expected_absence_count == 2
    assert node.change_events[-1]["type"] == "expected_object_missing"
    assert loaded.get_change_events()
    assert loaded._relation_beliefs
