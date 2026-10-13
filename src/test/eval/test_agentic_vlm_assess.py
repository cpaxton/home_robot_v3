# Copyright (c) Chris Paxton 2026

"""Unit tests for VLM-first target extract + view assess helpers."""

from __future__ import annotations

import json

import numpy as np

from emet.eval.agentic_vlm_assess import (
    assess_view_with_vlm,
    build_inventory_brief,
    extract_target_from_question,
    unique_image_arrays,
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


def test_assessment_keeps_current_and_context_roles_and_image_budget():
    client = _MultiClient({"present": False, "answerable": False})
    current = np.full((8, 8, 3), 10, dtype=np.uint8)
    context = np.full_like(current, 20)
    crops = [np.full_like(current, i) for i in (30, 40, 50)]
    result = assess_view_with_vlm(
        client,
        question="Where is the cup?",
        rgb=current,
        context_rgb=context,
        is_mcq=False,
        multi_close_look_crops=crops,
    )
    prompt, kwargs = client.mm_calls[-1]
    assert len(kwargs["image"]) == result.image_count == 4
    np.testing.assert_array_equal(kwargs["image"][0], current)
    np.testing.assert_array_equal(kwargs["image"][-1], context)
    assert "Judge present from Image 1 only" in prompt
    assert "earlier pre-approach context" in prompt
    assert not result.present and not result.answerable
    assess_view_with_vlm(client, question="Where is the cup?", rgb=current, context_rgb=current.copy())
    assert isinstance(client.mm_calls[-1][1]["image"], np.ndarray)


def test_selected_history_does_not_authorize_current_localization():
    from emet.memory.view_selection import ViewFrame, select_views

    frames = [ViewFrame(str(i), "episode", float(i), np.full((8, 8, 3), i, dtype=np.uint8)) for i in range(2)]
    selected = select_views(frames, current_id="1")
    client = _MultiClient({"present": False, "answerable": False})
    for is_mcq in (False, True):
        assess_view_with_vlm(
            client,
            question="Where is the cup?",
            rgb=frames[-1].rgb,
            selected_views=selected,
            is_mcq=is_mcq,
        )
        prompt, kwargs = client.mm_calls[-1]
        assert len(kwargs["image"]) == 2
        assert "Judge present from Image 1 only" in prompt
        assert ("Answerability may use the labeled historical context" in prompt) == is_mcq
        assert ("answerable requires the target to be visible in Image 1" in prompt) != is_mcq


def test_assessment_uses_exact_selected_packet_and_historical_labels():
    from emet.memory.view_selection import ViewFrame, select_views

    frames = [ViewFrame(str(i), "episode", float(i), np.full((8, 8, 3), i, dtype=np.uint8)) for i in range(4)]
    selected = select_views(frames, current_id="3")
    client = _MultiClient({"present": False, "answerable": False})
    assessment = assess_view_with_vlm(client, question="Where is the cup?", rgb=frames[-1].rgb, selected_views=selected)
    prompt, kwargs = client.mm_calls[-1]
    assert assessment.image_count == 4
    for actual, expected in zip(kwargs["image"], selected.views, strict=True):
        np.testing.assert_array_equal(actual, expected.frame.rgb)
    assert "Image 4" in prompt and "earlier context" in prompt
    assert "pre-approach" not in prompt
    assert "Judge present from Image 1 only" in prompt


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


def test_assess_view_accepts_legacy_labeled_letter():
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
    prompt = client.mm_calls[0][0]
    assert "exact semantic option text" in prompt
    assert "MCQ letter A-D" not in prompt


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


def test_assess_view_includes_siglip_evidence_in_prompt():
    """Image-text similarity for the target must reach the VLM (cube vs brick help)."""
    client = _MultiClient(
        {
            "target": "sugar cube",
            "present": True,
            "answerable": True,
            "need_more_views": False,
            "suggested_answer": None,
            "reason": "cube on floor visible",
        }
    )
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    out = assess_view_with_vlm(
        client,
        question="Where is the sugar cube on the floor?",
        rgb=rgb,
        inventory="observations_seen=5",
        target_phrase="sugar cube",
        siglip_evidence="'sugar cube' similarity=0.21 (present-like)",
    )
    assert out.present is True
    assert client.mm_calls
    prompt = client.mm_calls[0][0]
    assert "Visual evidence (image-text similarity)" in prompt
    assert "sugar cube' similarity=0.21" in prompt


def test_assess_view_without_evidence_omits_line():
    client = _MultiClient(
        {
            "target": "lamp",
            "present": False,
            "answerable": False,
            "need_more_views": True,
            "suggested_answer": None,
            "reason": "not visible",
        }
    )
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    assess_view_with_vlm(
        client,
        question="Where is the lamp?",
        rgb=rgb,
        target_phrase="lamp",
    )
    prompt = client.mm_calls[0][0]
    assert "Visual evidence" not in prompt


def test_assess_view_feeds_close_look_crop_as_second_image():
    """A close-look crop must reach the VLM as a second image (count/clock detail)."""
    client = _MultiClient(
        {
            "target": "clock",
            "present": True,
            "answerable": True,
            "need_more_views": False,
            "suggested_answer": None,
            "reason": "clock face readable in zoom",
        }
    )
    wide = np.zeros((16, 16, 3), dtype=np.uint8)
    crop = np.full((6, 6, 3), 128, dtype=np.uint8)
    out = assess_view_with_vlm(
        client,
        question="What time is it now?",
        rgb=wide,
        target_phrase="clock",
        close_look_crop=crop,
    )
    assert out.present is True
    assert client.mm_calls
    prompt, kwargs = client.mm_calls[0]
    assert "zoomed crop" in prompt
    imgs = kwargs.get("image")
    assert isinstance(imgs, list) and len(imgs) == 2
    assert imgs[1].shape == (6, 6, 3)


def test_assess_view_feeds_multi_close_look_crops():
    """Multiple close-look crops from different views must all reach the VLM."""
    client = _MultiClient(
        {
            "target": "pillows",
            "present": True,
            "answerable": True,
            "need_more_views": False,
            "suggested_answer": None,
            "reason": "counted across views",
        }
    )
    wide = np.zeros((16, 16, 3), dtype=np.uint8)
    crop_a = np.full((6, 6, 3), 10, dtype=np.uint8)
    crop_b = np.full((6, 6, 3), 200, dtype=np.uint8)
    out = assess_view_with_vlm(
        client,
        question="How many pillows?",
        rgb=wide,
        target_phrase="pillows",
        close_look_crop=crop_a,
        multi_close_look_crops=[crop_b],
    )
    assert out.present is True
    prompt, kwargs = client.mm_calls[0]
    assert "different views" in prompt
    imgs = kwargs.get("image")
    assert isinstance(imgs, list) and len(imgs) == 3  # wide + crop_a + crop_b
    assert imgs[1].shape == (6, 6, 3)
    assert imgs[2].shape == (6, 6, 3)


def test_assess_view_deduplicates_and_caps_close_look_crops():
    client = _MultiClient(
        {
            "target": "clock",
            "present": True,
            "answerable": True,
            "need_more_views": False,
            "suggested_answer": None,
            "reason": "readable",
        }
    )
    wide = np.zeros((16, 16, 3), dtype=np.uint8)
    crop_a = np.full((6, 6, 3), 10, dtype=np.uint8)
    crop_b = np.full((6, 6, 3), 20, dtype=np.uint8)
    crop_c = np.full((6, 6, 3), 30, dtype=np.uint8)
    crop_d = np.full((6, 6, 3), 40, dtype=np.uint8)

    assess_view_with_vlm(
        client,
        question="What time is it?",
        rgb=wide,
        target_phrase="clock",
        close_look_crop=crop_a,
        multi_close_look_crops=[crop_b, crop_b, crop_c, crop_d],
    )

    _prompt, kwargs = client.mm_calls[0]
    imgs = kwargs["image"]
    assert len(imgs) == 4  # wide + three unique close-look crops
    assert [int(img[0, 0, 0]) for img in imgs[1:]] == [10, 20, 30]
    assert len(unique_image_arrays([crop_a, crop_b, crop_b, crop_c, crop_d], max_images=3)) == 3
