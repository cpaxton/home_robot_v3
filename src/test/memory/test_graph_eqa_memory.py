# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Tests for GraphEQA memory (graph-based EQA). No code copied from closed-source repos.

import numpy as np
import pytest
from PIL import Image

from emet.memory.graph_eqa import GraphEQAMemory
from emet.memory.graph_eqa.graph_memory import _near, _on_floor


def test_graph_memory_add_observation():
    """Adding observations creates nodes and updates edges."""
    mem = GraphEQAMemory(
        eqa_client=lambda x: "reasoning: r\nanswer: yes\nconfidence: true\naction:\nconfidence_reasoning: ok",
        image_description_client=lambda x: "table, cup",
    )
    rgb = np.zeros((60, 80, 3), dtype=np.uint8)
    xyz1 = np.array([0.0, 0.0, 0.5])
    id1 = mem.add_observation(rgb, xyz1, ["table"])
    assert id1 == 1
    assert len(mem.get_nodes()) == 1
    assert len(mem.get_observations()) == 1

    id2 = mem.add_observation(rgb, np.array([0.3, 0.0, 0.5]), ["cup"])
    assert id2 == 2
    assert len(mem.get_nodes()) == 2
    edges = mem.get_edges()
    # near(table, cup) should exist
    assert any((1, 2, "near") == e or (2, 1, "near") == e for e in edges)


def test_graph_memory_to_string():
    """Scene graph serializes to a string for prompts."""
    mem = GraphEQAMemory(
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    rgb = np.zeros((60, 80, 3), dtype=np.uint8)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.0]), ["floor", "carpet"])
    mem.add_observation(rgb, np.array([0.5, 0.0, 0.8]), ["table"])
    s = mem.to_string()
    assert "SCENE_GRAPH" in s
    assert "floor" in s or "carpet" in s
    assert "table" in s
    assert "Node 1" in s
    assert "Node 2" in s


def test_parse_answer():
    """parse_answer extracts reasoning, answer, confidence, action, confidence_reasoning."""
    mem = GraphEQAMemory(
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    raw = (
        "reasoning: I see a table.\n"
        "answer: Yes\n"
        "confidence: True\n"
        "action: \n"
        "confidence_reasoning: I am sure."
    )
    r, a, c, act, cr = mem.parse_answer(raw)
    assert "table" in r
    assert a.strip() == "yes"
    assert c is True
    assert "sure" in cr.lower()


def test_parse_answer_not_confident():
    """When confidence is False, action can be an image id."""
    mem = GraphEQAMemory(
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    raw = (
        "reasoning: Need to look more.\n"
        "answer: Unknown\n"
        "confidence: FALSE\n"
        "action: 3\n"
        "confidence_reasoning: Not seen yet."
    )
    r, a, c, act, cr = mem.parse_answer(raw)
    assert c is False
    assert act.strip() == "3"


def test_near_heuristic():
    """_near returns True when 2D distance <= max_dist."""
    assert _near(np.array([0, 0, 0]), np.array([0.5, 0, 0]), max_dist=1.0) is True
    assert _near(np.array([0, 0, 0]), np.array([2, 0, 0]), max_dist=1.0) is False


def test_on_floor_heuristic():
    """_on_floor returns True when z <= threshold."""
    assert _on_floor(np.array([0, 0, 0.02])) is True
    assert _on_floor(np.array([0, 0, 0.2])) is False


def test_query_answer_returns_tuple_with_mock_client():
    """query_answer returns (reasoning, answer, confidence, confidence_reasoning, target_point, relevant_images)."""
    def mock_eqa(commands):
        return (
            "reasoning: I see a table.\n"
            "answer: Yes\n"
            "confidence: true\n"
            "action: \n"
            "confidence_reasoning: Sure."
        )

    mem = GraphEQAMemory(
        eqa_client=mock_eqa,
        image_description_client=lambda x: "table",
    )
    mem.add_observation(
        np.zeros((60, 80, 3), dtype=np.uint8),
        np.array([0.0, 0.0, 0.5]),
        ["table"],
    )
    out = mem.query_answer("Is there a table?", None, None)
    assert len(out) == 6
    reasoning, answer, confidence, confidence_reasoning, target_point, relevant_images = out
    assert isinstance(reasoning, str)
    assert isinstance(answer, str)
    assert isinstance(confidence, bool)
    assert isinstance(confidence_reasoning, str)
    assert target_point is None  # confident, so no exploration
    assert isinstance(relevant_images, list)
