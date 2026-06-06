# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for human-readable GraphEQA answer formatting."""

from __future__ import annotations

import numpy as np

from emet.memory.graph_eqa.graph_memory import GraphEQAMemory, GraphNode
from emet.memory.graph_eqa.human_answer import (
    HumanEQAResult,
    answer_looks_like_image_index,
    format_eqa_tool_response,
    format_human_eqa_answer,
)


def test_answer_looks_like_image_index():
    assert answer_looks_like_image_index("image 1")
    assert answer_looks_like_image_index("1", "where is the sink?")
    assert not answer_looks_like_image_index("The sink is on the counter.")


def test_format_human_eqa_from_image_id_uses_graph_node():
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem._nodes.append(
        GraphNode(
            node_id=1,
            labels=["sink"],
            xyz=np.array([2.1, -0.8, 0.9]),
            obs_id=1,
        )
    )
    human = format_human_eqa_answer(
        "Where is the sink?",
        "image 1",
        "Image 1 contains a sink.",
        mem,
        confidence=True,
    )
    assert "sink" in human.user_answer.lower()
    assert "2.10" in human.user_answer or "2.1" in human.user_answer
    assert not answer_looks_like_image_index(human.user_answer)


def test_format_eqa_tool_response_shape():
    text = format_eqa_tool_response(
        HumanEQAResult(
            user_answer="The sink is on the counter.",
            location_hint="(2.1, -0.8, 0.9) m",
            confidence_summary="confident",
            debug_reasoning="internal",
        )
    )
    assert text.startswith("Answer:")
    assert "Location:" in text
    assert "Confidence: confident" in text
    assert "image 1" not in text.lower()
