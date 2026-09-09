# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tests for merged-memory prompt mode (CONFIRMED_MEMORY folded into SCENE_GRAPH)."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np

from emet.memory.graph_eqa import GraphEQAMemory
from emet.memory.graph_eqa.graph_memory import SIGLIP_CONFIRM_THRESHOLD, SIGLIP_PRESENT_THRESHOLD


def _default_mem(**kwargs):
    mem = GraphEQAMemory(
        eqa_client=lambda x: "reasoning: r\nanswer: A\nconfidence: true\naction:\nconfidence_reasoning: ok",
        image_description_client=lambda x: "pillow",
        **kwargs,
    )
    mem.memory_summary_enabled = True
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem.add_observation(rgb, np.array([4.1, -2.3, 0.5]), ["red pillow"])
    mem.add_observation(rgb, np.array([4.4, -2.0, 0.5]), ["red pillow"])
    mem.add_observation(rgb, np.array([4.0, -2.5, 0.5]), ["sofa"])
    mem._relevant_phrases = ["red pillow", "towel"]
    mem._relevant_objects = ["red", "sofa", "towel"]
    return mem


def test_to_string_flat_without_merge_flag():
    """Without ``merge_confirmed``, to_string stays flat even though the default is now on."""
    mem = _default_mem()
    s = mem.to_string(max_object_nodes=48)
    assert "CONFIRMED_MEMORY" not in s
    assert " inspect" not in s
    assert "Rooms:" not in s


def test_merged_memory_default_on():
    """The folded format is the default; config true/false still overrides it."""
    assert GraphEQAMemory(defer_llm_clients=True)._merged_memory_enabled() is True
    mem = GraphEQAMemory(defer_llm_clients=True, parameters={"eqa": {"merged_memory": True}})
    assert mem._merged_memory_enabled() is True
    mem2 = GraphEQAMemory(defer_llm_clients=True, parameters={"eqa": {"merged_memory": "yes"}})
    assert mem2._merged_memory_enabled() is True
    mem3 = GraphEQAMemory(defer_llm_clients=True, parameters={"eqa": {"merged_memory": True}})
    mem3.parameters = {"eqa": {"merged_memory": False}}
    assert mem3._merged_memory_enabled() is False


def test_merged_memory_env_override(monkeypatch):
    """EMET_EQA_MERGED_MEMORY=0 beats config true; =1 beats config false."""
    mem = GraphEQAMemory(defer_llm_clients=True, parameters={"eqa": {"merged_memory": True}})
    monkeypatch.setenv("EMET_EQA_MERGED_MEMORY", "0")
    assert mem._merged_memory_enabled() is False
    monkeypatch.setenv("EMET_EQA_MERGED_MEMORY", "")
    assert mem._merged_memory_enabled() is True
    mem2 = GraphEQAMemory(defer_llm_clients=True)
    monkeypatch.setenv("EMET_EQA_MERGED_MEMORY", "1")
    assert mem2._merged_memory_enabled() is True


def test_merged_memory_folds_summary_into_scene_graph(monkeypatch):
    """Node lines carry inspect tags; grounded phrases are not re-listed; tail has the rest."""
    monkeypatch.setenv("EMET_EQA_MERGED_MEMORY", "1")
    mem = _default_mem()
    s = mem.to_string(max_object_nodes=48, merge_confirmed=True)
    assert " inspect" in s  # red pillow nodes tagged as views to inspect
    assert "- red pillow:" not in s  # grounded phrase not duplicated in tail
    assert "- towel: not observed during exploration" in s  # unobserved phrase stays
    assert "CONFIRMED_MEMORY" in s  # tail header kept for greppability
    assert "Rooms:" in s  # compact rooms line appended
    assert "nearest:" in s  # sofa neighbor preserved from legacy summary


def test_merged_memory_query_answer_end_to_end(monkeypatch):
    """query_answer: merged mode replaces the standalone CONFIRMED_MEMORY block."""
    monkeypatch.setenv("EMET_EQA_MERGED_MEMORY", "1")
    captured: dict = {}

    def _client(cmds):
        captured["cmds"] = cmds
        return "reasoning: r\nanswer: A\nconfidence: true\naction:\nconfidence_reasoning: ok"

    mem = GraphEQAMemory(eqa_client=_client, image_description_client=lambda x: "pillow")
    mem.memory_summary_enabled = True
    mem._enrich_object_hints = ["towel"]  # survives extract_relevant_objects; red pillow from heuristics
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem.add_observation(rgb, np.array([4.1, -2.3, 0.5]), ["red pillow"])
    mem.add_observation(rgb, np.array([4.4, -2.0, 0.5]), ["red pillow"])
    mem.add_observation(rgb, np.array([4.0, -2.5, 0.5]), ["sofa"])
    mem.query_answer("Where did I leave the red pillow? A) bedroom B) kitchen C) office. Answer:")
    text = "\n".join(str(c) for c in captured["cmds"])
    assert "CONFIRMED_MEMORY" in text  # tail header
    assert " inspect" in text  # status tag on a node line
    assert "- red pillow:" not in text  # no duplicate summary line
    assert "- towel: not observed during exploration" in text


def test_merged_memory_not_merged_when_summary_disabled(monkeypatch):
    """memory_summary_enabled=False keeps the plain graph even with the flag on."""
    monkeypatch.setenv("EMET_EQA_MERGED_MEMORY", "1")
    mem = _default_mem()
    mem.memory_summary_enabled = False
    s = mem.to_string(max_object_nodes=48, merge_confirmed=True)
    assert " inspect" not in s
    assert "CONFIRMED_MEMORY" not in s
    assert "Rooms:" not in s


def test_merged_memory_attribute_question_skips_priors(monkeypatch):
    """On/off questions must not inject status tags or memory tails."""
    monkeypatch.setenv("EMET_EQA_MERGED_MEMORY", "1")
    captured: dict = {}

    def _client(cmds):
        captured["cmds"] = cmds
        return "reasoning: lamp looks off\nanswer: B\nconfidence: true\naction:\nconfidence_reasoning: image"

    mem = GraphEQAMemory(eqa_client=_client, image_description_client=lambda x: "lamp")
    mem.memory_summary_enabled = True
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["lamp", "sofa"])
    mem._relevant_phrases = ["lamp sofa off"]
    mem._relevant_objects = ["lamp"]
    mem.query_answer("Is the lamp turned off? A) Yes B) No. Answer:")
    text = "\n".join(str(c) for c in captured["cmds"])
    assert "CONFIRMED_MEMORY" not in text
    assert " inspect" not in text


def test_merged_memory_room_tags(monkeypatch):
    """Nodes in a stamped room cluster get the room name inline."""
    monkeypatch.setenv("EMET_EQA_MERGED_MEMORY", "1")
    mem = _default_mem()
    mem.refresh_room_clusters()
    assert mem._room_clusters, "expected at least one room cluster"
    mem._room_clusters = [
        replace(c, room_name="kitchen" if 1 in c.node_ids else c.room_name) for c in mem._room_clusters
    ]
    s = mem.to_string(max_object_nodes=48, merge_confirmed=True)
    assert "Node 1 (kitchen):" in s
    assert "Rooms: kitchen(" in s


def test_merged_memory_siglip_only_is_candidate_not_present(monkeypatch):
    """High SigLIP alone must not claim present — that reintroduces detector-as-fact."""
    monkeypatch.setenv("EMET_EQA_MERGED_MEMORY", "1")
    mem = _default_mem()
    mem._relevant_phrases = ["woven basket"]  # no graph label match
    mem._relevant_objects = ["basket"]

    def _fake_sig(phrase: str):
        assert "basket" in phrase.lower() or phrase == "woven basket"
        return (float(SIGLIP_CONFIRM_THRESHOLD + 0.05), np.array([1.0, 2.0, 0.5]), 1)

    with patch.object(mem, "_siglip_match_for_phrase", side_effect=_fake_sig):
        s = mem.to_string(max_object_nodes=48, merge_confirmed=True)
    assert "woven basket: CANDIDATE" in s
    assert "present (SigLIP" not in s
    assert "woven basket: present" not in s
    assert "do not treat as confirmed present or absent" in s


def test_merged_memory_weak_siglip_does_not_assert_absence(monkeypatch):
    """Low SigLIP must not say 'likely NOT present' (ABSENT coloring)."""
    monkeypatch.setenv("EMET_EQA_MERGED_MEMORY", "1")
    mem = _default_mem()
    mem._relevant_phrases = ["wall clock"]
    mem._relevant_objects = ["clock"]

    def _fake_sig(phrase: str):
        return (float(SIGLIP_PRESENT_THRESHOLD - 0.05), np.array([0.0, 0.0, 0.5]), 1)

    with patch.object(mem, "_siglip_match_for_phrase", side_effect=_fake_sig):
        s = mem.to_string(max_object_nodes=48, merge_confirmed=True)
    assert "likely NOT present" not in s
    assert "ABSENT" not in s
    assert "weak SigLIP only" in s
    assert "not evidence of absence" in s


def test_merged_memory_outside_budget_preserves_present(monkeypatch):
    """Graph-grounded present nodes truncated from the dump stay in the tail with facts.

    The tail must carry coordinates + count + nearest furniture (legacy parity),
    not dangling node ids the model cannot see.
    """
    monkeypatch.setenv("EMET_EQA_MERGED_MEMORY", "1")
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.memory_summary_enabled = True
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem.add_observation(rgb, np.array([4.1, -2.3, 0.5]), ["cup"])  # keyword match
    mem.add_observation(rgb, np.array([4.4, -2.0, 0.5]), ["cup"])  # keyword match
    mem.add_observation(rgb, np.array([1.0, 1.0, 0.5]), ["towel rack"])  # phrase match, not kw
    mem.add_observation(rgb, np.array([0.8, 1.2, 0.5]), ["sofa"])  # nearest to towel rack
    mem._relevant_phrases = ["towel"]
    mem._relevant_objects = ["cup", "towel"]
    # Budget 1: keyword matches win the slot; the towel-rack node is fully truncated.
    s = mem.to_string(max_object_nodes=1, merge_confirmed=True, question_keywords=["cup"])
    assert "- towel: LOOK" in s
    assert "[graph obs 3] at (1.0, 1.0)" in s
    assert "towel rack at (1.0, 1.0)" not in s
    assert "1 graph node(s) at (1.0, 1.0)" not in s
    assert "nearest: sofa at (0.8, 1.2)" in s
    assert "nodes not shown in graph above" in s
    assert "towel: not observed" not in s
    assert "towel: CANDIDATE" not in s
