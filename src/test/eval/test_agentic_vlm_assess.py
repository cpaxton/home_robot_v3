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
    client = _TextClient(
        {"target_phrase": "utensils", "question_type": "location", "notes": "ignore options"}
    )
    te = extract_target_from_question(
        client,
        "Where are the utensils already set? A) table B) counter",
        fallback_phrase="sets utensils already",
    )
    assert te.target_phrase == "utensils"
    assert te.question_type == "location"
    assert client.calls


def test_extract_target_falls_back_on_empty():
    client = _TextClient({})
    te = extract_target_from_question(client, "Where is the lamp?", fallback_phrase="lamp")
    assert te.target_phrase == "lamp"


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


def test_build_inventory_brief_includes_proposal():
    brief = build_inventory_brief(
        n_observations=3,
        graph_labels=["sink", "towel"],
        proposal={"phrase": "sink", "detector_score": 0.2, "obs_id": 4},
        tried_obs_ids=[1, 4],
        n_rounds=2,
        n_nav=1,
    )
    assert "observations_seen=3" in brief
    assert "sink" in brief
    assert "detector_score=0.2" in brief
