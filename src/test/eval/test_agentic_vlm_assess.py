# Copyright (c) Chris Paxton 2026

"""Unit tests for VLM-first target extract + view assess helpers."""

from __future__ import annotations

import json

import numpy as np

from emet.eval.agentic_vlm_assess import (
    assess_view_with_vlm,
    build_inventory_brief,
    extract_target_from_question,
)


class _TextClient:
    def __init__(self, payload: dict):
        self._payload = payload
        self.calls: list = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        return json.dumps(self._payload)


class _MultiClient:
    def __init__(self, payload: dict):
        self._vl = self
        self._payload = payload
        self.mm_calls: list = []

    def generate_multimodal(self, prompt, **kwargs):
        self.mm_calls.append((prompt, kwargs))
        return json.dumps(self._payload)

    def __call__(self, command, **kwargs):
        return json.dumps(self._payload)


def test_extract_target_from_question_parses_json():
    client = _TextClient({"target_phrase": "utensils", "question_type": "location", "notes": "ignore options"})
    extracted = extract_target_from_question(
        client,
        "Where are the utensils already set? A) table B) counter",
        fallback_phrase="sets utensils already",
    )
    assert extracted.target_phrase == "utensils"
    assert extracted.question_type == "location"
    assert client.calls


def test_extract_target_falls_back_on_empty():
    client = _TextClient({})
    extracted = extract_target_from_question(client, "Where is the lamp?", fallback_phrase="lamp")
    assert extracted.target_phrase == "lamp"


def test_assess_view_uses_generate_multimodal():
    client = _MultiClient(
        {
            "target": "air conditioning",
            "present": True,
            "answerable": False,
            "need_more_views": True,
            "suggested_answer": None,
            "reason": "AC visible but on/off unclear",
        }
    )
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    out = assess_view_with_vlm(
        client,
        question="Is the AC on?",
        rgb=rgb,
        inventory="observations_seen=2",
        target_phrase="air conditioning",
    )
    assert out.present is True
    assert out.answerable is False
    assert out.need_more_views is True
    assert client.mm_calls


def test_assess_view_parses_string_booleans_and_labeled_letter():
    client = _MultiClient(
        {
            "target": "clock",
            "present": "false",
            "answerable": "true",
            "need_more_views": "false",
            "suggested_answer": "Answer: C",
        }
    )
    out = assess_view_with_vlm(
        client,
        question="What time is it? A) Morning B) Noon C) Evening D) Night",
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        target_phrase="clock",
    )
    assert out.present is False
    assert out.answerable is True
    assert out.need_more_views is False
    assert out.suggested_answer == "C"


def test_assess_view_does_not_turn_choice_text_into_first_letter():
    client = _MultiClient(
        {
            "target": "vase",
            "present": True,
            "answerable": True,
            "need_more_views": False,
            "suggested_answer": "Dining room",
        }
    )
    out = assess_view_with_vlm(
        client,
        question="Where is the vase? A) Kitchen B) Bedroom C) Patio D) Dining room",
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        target_phrase="vase",
    )
    assert out.suggested_answer == "Dining room"


def test_build_inventory_brief_excludes_detector_proposal():
    """SigLIP/OWL verdicts must not enter assess inventory (they color answers)."""
    brief = build_inventory_brief(
        n_observations=3,
        graph_labels=["sink", "towel"],
        tried_obs_ids=[1, 4],
        n_rounds=2,
        n_nav=1,
    )
    assert "observations_seen=3" in brief
    assert "sink" in brief
    assert "cheap_proposal" not in brief
    assert "decision=" not in brief
    assert "ABSENT" not in brief
