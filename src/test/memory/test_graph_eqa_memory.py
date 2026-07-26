# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Tests for GraphEQA memory (graph-based EQA). No code copied from closed-source repos.

import tempfile
from pathlib import Path

import numpy as np

from emet.memory.adapters import GraphEQABackend
from emet.memory.graph_eqa import GraphEQAMemory, labels_are_semantic_graph_hypothesis
from emet.memory.graph_eqa.graph_memory import (
    GraphNode,
    GraphObservation,
    _near,
    _on_floor,
    consolidate_relevant_keywords,
    heuristic_relevant_phrases,
    label_matches_relevant_object,
)


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
    raw = "reasoning: I see a table.\nanswer: Yes\nconfidence: True\naction: \nconfidence_reasoning: I am sure."
    r, a, c, act, cr = mem.parse_answer(raw)
    assert "table" in r
    assert a.strip().lower() == "yes"
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


def test_relevant_memory_summary_surfaces_observed_objects():
    """CONFIRMED_MEMORY lists relevant objects present via graph nodes; flags missing ones."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([4.1, -2.3, 0.5]), ["red pillow"])
    mem.add_observation(rgb, np.array([4.4, -2.0, 0.5]), ["red pillow"])
    mem.add_observation(rgb, np.array([4.0, -2.5, 0.5]), ["sofa"])

    mem._relevant_objects = ["red", "sofa"]
    summary = mem._relevant_memory_summary()
    assert "CONFIRMED_MEMORY" in summary
    assert "red: PRESENT" in summary and "2 graph node(s)" in summary
    assert "sofa: PRESENT" in summary

    mem._relevant_objects = ["unicorn"]
    missing = mem._relevant_memory_summary()
    assert "unicorn: not observed during exploration" in missing


def test_relevant_memory_summary_includes_nearest_furniture():
    """Location MCQ helper: PRESENT lines list nearby furniture labels."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([-1.77, 4.11, 0.5]), ["basket", "cabinet"])
    mem.add_observation(rgb, np.array([-0.71, 3.12, 0.5]), ["armchair"])
    mem.add_observation(rgb, np.array([10.0, 10.0, 0.5]), ["oven"])  # far away
    mem._relevant_phrases = ["woven basket"]
    mem._relevant_objects = ["basket"]
    summary = mem._relevant_memory_summary()
    assert "woven basket: PRESENT" in summary or "basket: PRESENT" in summary
    assert "nearest:" in summary
    assert "armchair" in summary
    assert "oven" not in summary.split("nearest:")[-1]  # far oven not among nearest


def test_location_letter_from_nearest_memory_picks_armchair_option():
    """Nearest armchair maps to the living-room-armchairs MCQ letter."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([-1.77, 4.11, 0.5]), ["basket", "cabinet"])
    mem.add_observation(rgb, np.array([-0.71, 3.12, 0.5]), ["armchair"])
    mem._relevant_phrases = ["woven basket"]
    mem._relevant_objects = ["basket"]
    choices = [
        "By the kitchen counter",
        "Between TV and living room sofas",
        "Next to the dining table",
        "Next to the living room armchairs",
    ]
    assert mem._location_letter_from_nearest_memory(choices) == "D"


def test_label_matches_trash_can_to_recycle_bin():
    assert label_matches_relevant_object("silver trash can", "recycle bin")
    assert label_matches_relevant_object("trash can", "garbage bin")


def test_select_relevant_obs_ids_prefers_choice_landmarks_before_siglip():
    """Location MCQ: refrigerator landmark beats a SigLIP dining-table view for Image 1."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["dining table", "chair"])
    mem.add_observation(rgb, np.array([2.0, 1.0, 0.5]), ["refrigerator", "recycle bin"])
    mem._relevant_objects = ["trash can"]
    mem._relevant_phrases = ["silver trash can"]
    # Fake SigLIP pointing at the dining table observation.
    mem._siglip_phrase_cache["silver trash can"] = (0.35, np.array([0.0, 0.0, 0.5]), 1)
    choices = [
        "Next to the dining table",
        "Next to the TV",
        "Next to the kitchen sink",
        "Next to the refrigerator",
    ]
    obs_ids = mem._select_relevant_obs_ids(max_images=3, choices=choices)
    assert obs_ids[0] == 2  # refrigerator / recycle bin (boosted over dining table)


def test_select_relevant_obs_ids_prefers_fridge_over_recycle_alias():
    """Trash keyword→recycle-bin alias must not beat refrigerator choice landmark."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["recycle bin"])
    mem.add_observation(rgb, np.array([1.0, 1.0, 0.5]), ["refrigerator"])
    mem._relevant_objects = ["trash can"]
    mem._relevant_phrases = ["silver trash can"]
    choices = [
        "Next to the dining table",
        "Next to the TV",
        "Next to the kitchen sink",
        "Next to the refrigerator",
    ]
    obs_ids = mem._select_relevant_obs_ids(max_images=3, choices=choices)
    assert obs_ids[0] == 2  # refrigerator, not recycle bin


def test_select_relevant_obs_ids_prefers_fridge_over_sink_for_trash():
    """Sink must not beat refrigerator for trash Image 1 (competing MCQ option)."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["sink", "microwave", "cabinet"])
    mem.add_observation(rgb, np.array([1.0, 1.0, 0.5]), ["refrigerator", "recycle bin"])
    mem._relevant_objects = ["trash can"]
    mem._relevant_phrases = ["silver trash can"]
    choices = [
        "Next to the dining table",
        "Next to the TV",
        "Next to the kitchen sink",
        "Next to the refrigerator",
    ]
    obs_ids = mem._select_relevant_obs_ids(max_images=3, choices=choices)
    assert obs_ids[0] == 2


def test_select_relevant_obs_ids_prefers_target_over_generic_choice():
    """Ladder view must beat dining-table choice landmark for Image 1."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["table", "fireplace", "stove"])
    mem.add_observation(rgb, np.array([1.0, 1.0, 0.5]), ["ladder", "sofa"])
    mem._relevant_objects = ["ladder"]
    mem._relevant_phrases = ["ladder"]
    choices = [
        "By the dining table",
        "Next to the living room dressor",
        "In the bathroom",
        "Next to TV",
    ]
    obs_ids = mem._select_relevant_obs_ids(max_images=3, choices=choices)
    assert obs_ids[0] == 2  # ladder, not table/fireplace


def test_select_relevant_obs_ids_attribute_prefers_lamp_over_frontier():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["frontier"])
    # Mark obs 1 as frontier node
    mem._nodes[0].is_frontier = True
    mem.add_observation(rgb, np.array([1.0, 1.0, 0.5]), ["lamp", "sofa"])
    mem._relevant_objects = ["lamp"]
    mem._relevant_phrases = ["lamp"]
    obs_ids = mem._select_relevant_obs_ids(max_images=2, attribute_question=True)
    assert obs_ids[0] == 2


def test_location_letter_prefers_image_landmarks_over_siglip_nearest():
    """Attached fridge view wins over SigLIP-nearest dining table for trash MCQ."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    raw = (
        "reasoning: memory says dining table\nanswer: A\nconfidence: true\n"
        "action: none\nconfidence_reasoning: memory"
    )
    mem = GraphEQAMemory(eqa_client=lambda _c: raw, image_description_client=lambda _x: "table")
    mem.memory_summary_enabled = True
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["dining table"])
    mem.add_observation(rgb, np.array([2.0, 1.0, 0.5]), ["refrigerator", "recycle bin"])
    mem._relevant_phrases = ["silver trash can"]
    mem._relevant_objects = ["trash can"]
    mem._siglip_phrase_cache["silver trash can"] = (0.35, np.array([0.0, 0.0, 0.5]), 1)
    q = (
        "Where did I leave the silver trash can at? "
        "A) Next to the dining table B) Next to the TV "
        "C) Next to the kitchen sink D) Next to the refrigerator. Answer:"
    )
    _r, answer, confidence, _cr, _pt, _imgs = mem.query_answer(q)
    assert answer.strip().upper().startswith("D") or "D" in answer.upper()
    # SigLIP-only target: do not finalize until images support the letter.
    assert confidence is False


def test_query_answer_does_not_finalize_under_equipment_without_geometry():
    """Under-X MCQs stay non-confident until mat↔equipment distance is known."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    raw = (
        "reasoning: bike nearby\nanswer: B\nconfidence: true\n"
        "action: none\nconfidence_reasoning: guess bike"
    )
    mem = GraphEQAMemory(eqa_client=lambda _c: raw, image_description_client=lambda _x: "bike")
    mem.memory_summary_enabled = True
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["stationary bike", "treadmill"])
    mem._relevant_phrases = ["exercise mat"]
    mem._relevant_objects = ["mat"]
    _r, _a, confidence, _cr, _pt, _imgs = mem.query_answer(
        "Did I leave the exercise mat under any workout equipment? "
        "A) Yes, under the elliptical machine B) Yes, under the stationary bike "
        "C) No, it's not under any workout equipment D) Yes, under the treadmill. Answer:"
    )
    assert confidence is False


def test_query_answer_does_not_finalize_location_without_target_view():
    """Location MCQ must not confirm when the target object is absent from attached images."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    raw = (
        "reasoning: guess dining table\nanswer: A\nconfidence: true\n"
        "action: none\nconfidence_reasoning: guess"
    )
    mem = GraphEQAMemory(eqa_client=lambda _c: raw, image_description_client=lambda _x: "table")
    mem.memory_summary_enabled = True
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["dining table", "chair"])
    mem._relevant_phrases = ["striped towel"]
    mem._relevant_objects = ["towel"]
    _r, _a, confidence, _cr, _pt, _imgs = mem.query_answer(
        "Where did I leave the striped towel? "
        "A) On the dining table B) On the living room floor "
        "C) In the bathroom D) By the kitchen sink. Answer:"
    )
    assert confidence is False
    # Either graph-cover or target-visibility gate should block finalization.


def test_memory_location_does_not_override_clear_vlm_letter():
    """Nearest-furniture memory must not clobber a clear VLM letter (ladder→B)."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    raw = (
        "reasoning: ladder visible in living room\nanswer: B\nconfidence: true\n"
        "action: none\nconfidence_reasoning: images show ladder by dresser"
    )
    mem = GraphEQAMemory(eqa_client=lambda _c: raw, image_description_client=lambda _x: "ladder")
    mem.memory_summary_enabled = True
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["ladder"])
    mem._relevant_phrases = ["ladder"]
    mem._relevant_objects = ["ladder"]
    # Simulate: images give no unique landmark letter, memory would prefer A.
    mem._location_letter_from_attached_images = lambda _choices, _obs_ids: ""  # type: ignore[method-assign]
    mem._location_letter_from_nearest_memory = lambda _choices: "A"  # type: ignore[method-assign]
    q = (
        "Where is the ladder? "
        "A) By the dining table B) Next to the living room dressor "
        "C) In the bathroom D) Next to TV. Answer:"
    )
    _r, answer, _c, _cr, _pt, _imgs = mem.query_answer(q)
    assert mem.last_eqa_parsed[1].strip().lower() == "b"
    assert "[memory-location]" not in (mem.last_eqa_raw or "")
    assert "B" in answer.upper() or "dresser" in answer.lower() or "ladder" in answer.lower()


def test_memory_location_does_not_override_vlm_choice_text():
    """Holdout q56 regression: NL answer matching choice C must not become memory A."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    raw = (
        "Caption:\nImage 1 shows a large black bookshelf and blue curtains.\n"
        "Reasoning:\nMatches option C.\n"
        "Answer:\nThe room with the blue curtains.\n"
        "Confidence:\nTRUE\n"
        "Action:\nNone\n"
        "Confidence_reasoning:\nImage 1 shows blue curtains.\n"
    )

    def _client(cmds, **_kw):
        return raw

    mem = GraphEQAMemory(eqa_client=_client, image_description_client=lambda _x: "bookshelf")
    mem.memory_summary_enabled = True
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["bookshelf", "chair"])
    mem._relevant_phrases = ["bookshelf"]
    mem._relevant_objects = ["bookshelf"]
    mem._location_letter_from_attached_images = lambda _choices, _obs_ids: ""  # type: ignore[method-assign]
    mem._location_letter_from_nearest_memory = lambda _choices: "A"  # type: ignore[method-assign]
    q = (
        "I need to retrieve a book from the shelf. Which room has the large bookshelf? "
        "A) The room with the white curtains B) The room with the yellow curtains "
        "C) The room with the blue curtains D) The room with the red curtains. Answer:"
    )
    _r, _answer, _c, _cr, _pt, _imgs = mem.query_answer(q)
    assert "[memory-location]" not in (mem.last_eqa_raw or "")
    assert mem.last_eqa_parsed[1].strip().upper() == "C"
    assert "[choice-text]" in (mem.last_eqa_raw or "")


def test_unknown_answer_triggers_letter_salvage():
    """Attribute/holdout q65 regression: answer Unknown must salvage a letter."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    calls = {"n": 0}

    def _client(cmds, **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return (
                "Caption:\nAC on ceiling.\nReasoning:\ncannot tell.\n"
                "Answer:\nUnknown\nConfidence:\nFALSE\nAction:\n1\n"
                "Confidence_reasoning:\nno status visible\n"
            )
        # Salvage turn — terse letter only.
        return "A"

    mem = GraphEQAMemory(eqa_client=_client, image_description_client=lambda _x: "ac")
    mem.memory_summary_enabled = True
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["air conditioner"])
    mem._relevant_phrases = ["air conditioner"]
    mem._relevant_objects = ["air conditioner"]
    q = (
        "Is the air conditioner turned on? "
        "A) On B) Off C) Unknown D) (Do not choose). Answer:"
    )
    _r, _answer, _c, _cr, _pt, _imgs = mem.query_answer(q)
    assert calls["n"] >= 2
    assert "[salvage]" in (mem.last_eqa_raw or "")
    assert mem.last_eqa_parsed[1].strip().upper() == "A"


def test_location_unknown_does_not_salvage_letter():
    """Holdout q104/q105: location Unknown must not invent A–D; keep Action explore."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    calls = {"n": 0}

    def _client(cmds, **_kw):
        calls["n"] += 1
        return (
            "Caption:\nno clock in view.\nReasoning:\nneed another room.\n"
            "Answer:\nUnknown\nConfidence:\nFALSE\nAction:\n1\n"
            "Confidence_reasoning:\nobject not visible\n"
        )

    mem = GraphEQAMemory(eqa_client=_client, image_description_client=lambda _x: "wall")
    mem.memory_summary_enabled = True
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["wall"])
    mem._relevant_phrases = ["clock"]
    mem._relevant_objects = ["clock"]
    # Include "On the …" options (q105) — must not be misclassified as attribute/on-off.
    q = (
        "I'm looking for the fruit bowl. "
        "A) On the kitchen island B) On the dining table "
        "C) On the coffee table D) In the sunroom. Answer:"
    )
    _r, answer, confidence, _cr, _pt, _imgs = mem.query_answer(q)
    assert calls["n"] == 1
    assert "[salvage]" not in (mem.last_eqa_raw or "")
    assert "[salvage-location]" not in (mem.last_eqa_raw or "")
    assert not confidence
    assert answer.strip().lower() in {"unknown", ""}
    assert mem.last_eqa_action_obs_id is not None


def test_location_truncated_stream_salvages_letter():
    """Dogfood q104/q105: a stream cut off before ``answer:`` must be re-asked.

    Distinct from an explicit ``Answer: Unknown`` (which stays Unknown): here the
    256-token decode budget ran out mid-caption, so the model never got to answer.
    """
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    calls = {"n": 0}

    def _client(cmds, **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            # Truncated mid-sentence: no "answer:" field ever emitted.
            return (
                "Caption:\nImage 1 shows an outdoor area with a brick path, door, "
                "doormat, glass door, greenery, outdoor furniture, pool, potted plant,"
            )
        return "D"

    mem = GraphEQAMemory(eqa_client=_client, image_description_client=lambda _x: "wall")
    mem.memory_summary_enabled = True
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["wall"])
    mem._relevant_phrases = ["wall clock"]
    mem._relevant_objects = ["wall clock"]
    q = (
        "I'm trying to remember where I placed the large wall clock. Where is it? "
        "A) In the dining area B) In the kitchen C) In the sunroom "
        "D) In the living area near the fireplace. Answer:"
    )
    _r, _answer, _c, _cr, _pt, _imgs = mem.query_answer(q)
    assert calls["n"] >= 2
    assert "[salvage]" in (mem.last_eqa_raw or "")
    assert mem.last_eqa_parsed[1].strip().upper() == "D"
    # Recovery must come from the VLM re-ask, never from nearest-furniture geometry.
    assert "[memory-location]" not in (mem.last_eqa_raw or "")


def test_verify_reports_unavailable_when_siglip_released():
    """submit_answer frees SigLIP for the VLM; later verifies are not evidence of absence."""
    mem = GraphEQAMemory(defer_llm_clients=True)
    oid = mem.add_observation(np.zeros((4, 4, 3), dtype=np.uint8), np.array([1.0, 1.0, 0.5]), ["wall"])
    mem.set_confirmed_memory_siglip_encoder(None)

    result = mem.verify_phrase_at_obs("fruit bowl", int(oid))
    assert result.status == "UNAVAILABLE"
    assert result.ok is False
    assert result.sim == 0.0


def test_location_truncated_empty_does_not_memory_location_letter():
    """failfix5: truncated stream with no answer: must not invent [memory-location] B.

    The nearest-furniture memory letter stays banned. Recovery is allowed only through
    a neutral VLM re-ask, so when that re-ask also declines the answer stays Unknown.
    """
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    calls = {"n": 0}

    def _client(cmds, **_kw):
        calls["n"] += 1
        if calls["n"] > 1:
            # Neutral re-ask still cannot tell — must not fall back to memory "B".
            return "I cannot tell from these images."
        # No ``answer:`` field — matches truncated Caption/Reasoning mid-choice.
        return (
            "Caption:\nkitchen and dining table.\n"
            "Reasoning:\nthe dining area (A) and kitchen (B) are candidates. The sunroom (C) "
            "and living area near the fireplace (\n"
            "Confidence:\nFALSE\n"
            "Action:\n1\n"
            "Confidence_reasoning:\nneed more views\n"
        )

    mem = GraphEQAMemory(eqa_client=_client, image_description_client=lambda _x: "clock")
    mem.memory_summary_enabled = True
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["dining table", "cabinet"])
    mem._relevant_phrases = ["large wall clock"]
    mem._relevant_objects = ["clock"]
    mem._any_confirmed_phrase_present = lambda: True  # type: ignore[method-assign]
    mem._location_letter_from_attached_images = lambda _c, _o: ""  # type: ignore[method-assign]
    mem._location_letter_from_nearest_memory = lambda _c: "B"  # type: ignore[method-assign]
    q = (
        "I'm trying to remember where I placed the large wall clock. Where is it? "
        "A) In the dining area B) In the kitchen C) In the sunroom "
        "D) In the living area near the fireplace. Answer:"
    )
    _r, answer, confidence, _cr, _pt, _imgs = mem.query_answer(q)
    assert "[memory-location]" not in (mem.last_eqa_raw or "")
    assert "[salvage-location]" not in (mem.last_eqa_raw or "")
    assert not confidence
    assert answer.strip().lower() == "unknown"
    assert mem.last_eqa_parsed[1].strip().lower() == "unknown"


def test_attribute_state_skips_memory_summary_block():
    """On/off questions should not inject CONFIRMED_MEMORY priors."""
    captured: dict = {}

    def _client(cmds):
        captured["cmds"] = cmds
        return (
            "reasoning: lamp looks off\nanswer: B\nconfidence: true\n"
            "action:\nconfidence_reasoning: image"
        )

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(eqa_client=_client, image_description_client=lambda _x: "lamp")
    mem.memory_summary_enabled = True
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["lamp", "sofa"])
    mem._relevant_phrases = ["lamp sofa off"]
    mem._relevant_objects = ["lamp"]
    mem.query_answer(
        "Is the lamp next to the sofa turned on or off? "
        "A) (Do not choose) B) Off C) On D) (Do not choose). Answer:"
    )
    assert not any(isinstance(c, str) and "CONFIRMED_MEMORY" in c for c in captured["cmds"])


def test_query_answer_does_not_finalize_yes_no_when_uncovered():
    """Uncovered relevant objects: Yes/No stays non-confident so exploration continues."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    raw = (
        "reasoning: no blanket in view\nanswer: No\nconfidence: true\n"
        "action: none\nconfidence_reasoning: none seen"
    )
    mem = GraphEQAMemory(eqa_client=lambda _c: raw, image_description_client=lambda _x: "sofa")
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["sofa"])
    mem._relevant_objects = ["blanket"]
    mem._relevant_phrases = ["blanket"]
    mem.memory_summary_enabled = True
    _r, _a, confidence, _cr, _pt, _imgs = mem.query_answer(
        "Is there a blanket on the bed? A) Yes B) No C) Maybe D) Unknown. Answer:"
    )
    assert confidence is False
    assert mem.last_eqa_model_confident is True


def test_select_relevant_obs_ids_prefers_keyword_target_before_grounder():
    """HM3D keyword/label matches should be Image 1 ahead of SigLIP grounder fills."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["oven"])
    mem.add_observation(rgb, np.array([1.0, 0.0, 0.5]), ["basket", "cabinet"])
    mem.add_observation(rgb, np.array([2.0, 0.0, 0.5]), ["door"])
    mem._relevant_objects = ["basket"]
    mem._relevant_phrases = ["woven basket"]
    # Grounder points at oven (wrong); keyword match for basket must still win Image 1.
    mem.set_obs_id_grounder(lambda text: 1)
    obs_ids = mem._select_relevant_obs_ids(max_images=3)
    assert obs_ids[0] == 2


def test_relevant_memory_summary_uses_siglip_grounder():
    """A SigLIP visual match marks an object PRESENT even with no caption-matched node."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    # Captioned as a plant, not a basket -> no graph node will match "basket".
    mem.add_observation(rgb, np.array([3.1, 5.0, 0.5]), ["decorative plant"])

    def grounder(text):
        if "basket" in text:
            return (0.30, np.array([3.1, 5.0, 0.5]))  # strong visual match
        return (0.05, np.array([0.0, 0.0, 0.0]))  # weak/no match

    mem.set_text_grounder(grounder)

    mem._relevant_phrases = ["woven basket"]
    mem._relevant_objects = ["woven", "basket"]
    summary = mem._relevant_memory_summary()
    assert "woven basket: PRESENT" in summary
    assert "SigLIP phrase match" in summary

    mem._relevant_phrases = ["elephant"]
    weak = mem._relevant_memory_summary()
    assert "elephant: likely NOT present" in weak
    assert "no strong SigLIP match" in weak


def test_relevant_memory_summary_uses_siglip_phrase_cache():
    """Cached graph-obs SigLIP alignments work after the voxel grounder is cleared."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.memory_summary_enabled = True
    oid = mem.add_observation(rgb, np.array([3.1, 5.0, 0.5]), ["decorative plant"])
    mem._relevant_phrases = ["woven basket"]
    mem._siglip_phrase_cache["woven basket"] = (0.32, np.array([3.1, 5.0, 0.5]), oid)
    mem._text_grounder = None

    summary = mem._relevant_memory_summary()
    assert "woven basket: PRESENT" in summary
    assert "obs_id=" in summary
    assert mem._graph_covers_relevant_objects()


def test_heuristic_relevant_phrases_prefers_multiword():
    phrases = heuristic_relevant_phrases("Did you see the woven basket anywhere?")
    assert phrases[0] == "woven basket"
    assert "anywhere" not in phrases


def test_heuristic_relevant_phrases_prefers_object_over_verb_stem():
    """Holdout q104/q105: do not verify SigLIP on ``trying remember placed``."""
    clock = heuristic_relevant_phrases(
        "I'm trying to remember where I placed the large wall clock. Where is it?"
    )
    assert clock[0] == "large wall clock"
    bowl = heuristic_relevant_phrases("I'm looking for the fruit bowl.")
    assert bowl[0] == "fruit bowl"
    from emet.memory.graph_eqa.graph_memory import heuristic_relevant_objects

    objs = heuristic_relevant_objects(
        "I'm trying to remember where I placed the large wall clock. Where is it?"
    )
    assert "clock" in objs
    assert "trying" not in objs
    assert "remember" not in objs


def test_heuristic_phrases_strip_mcq_options():
    """Agentic full question string must not yield ``table sunroom answer``."""
    from emet.memory.graph_eqa.graph_memory import heuristic_relevant_objects, question_stem_for_keywords

    full = (
        "I'm looking for the fruit bowl. "
        "A) On the kitchen island B) On the dining table "
        "C) On the coffee table D) In the sunroom. Answer:"
    )
    assert "fruit bowl" in question_stem_for_keywords(full).lower()
    assert "sunroom" not in question_stem_for_keywords(full).lower()
    phrases = heuristic_relevant_phrases(full)
    assert phrases[0] == "fruit bowl"
    assert "sunroom" not in " ".join(phrases)
    assert "answer" not in heuristic_relevant_objects(full)


def test_consolidate_relevant_keywords_drops_subsumed_tokens():
    phrases, objects = consolidate_relevant_keywords(
        ["woven basket"],
        ["woven", "basket", "anywhere", "kitchen"],
    )
    assert phrases == ["woven basket"]
    assert objects == ["woven basket", "kitchen"]


def test_extract_relevant_objects_prefers_phrases():
    mem = GraphEQAMemory(
        defer_llm_clients=True,
        image_description_client=lambda _x: "woven, basket, anywhere",
    )
    q = (
        "Did you see the woven basket anywhere? "
        "A) By the kitchen counter B) Between TV and living room sofas "
        "C) Next to the dining table D) Next to the living room armchairs. Answer:"
    )
    mem.extract_relevant_objects(q)
    assert mem._relevant_phrases == ["woven basket"]
    assert mem._relevant_objects[0] == "woven basket"
    assert "anywhere" not in mem._relevant_objects
    assert "woven" not in mem._relevant_objects or mem._relevant_objects[0] == "woven basket"


def test_query_answer_injects_location_mcq_hint():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    captured = {"cmds": None}

    def fake_eqa(cmds):
        captured["cmds"] = cmds
        return "reasoning: r\nanswer: d\nconfidence: true\naction: none\nconfidence_reasoning: x"

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=lambda _x: "basket",
    )
    mem.memory_summary_enabled = True
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["basket"])
    q = (
        "Did you see the woven basket anywhere? "
        "A) By the kitchen counter B) Between TV and living room sofas "
        "C) Next to the dining table D) Next to the living room armchairs. Answer:"
    )
    mem.extract_relevant_objects(q)
    mem.query_answer(q)
    assert any(isinstance(c, str) and "LOCATION_MCQ" in c for c in captured["cmds"])


def test_graph_covers_uses_phrase_not_every_token():
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem._relevant_phrases = ["woven basket"]
    mem._relevant_objects = ["woven", "basket", "anywhere"]
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["basket"])
    assert mem._graph_covers_relevant_objects()


def test_query_answer_caps_history_in_prompt():
    """Only the most recent eqa_max_history iterations are sent to the VLM."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    captured = {"cmds": None}

    def fake_eqa(cmds):
        captured["cmds"] = cmds
        return "reasoning: r\nanswer: B\nconfidence: false\naction: none\nconfidence_reasoning: x"

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=lambda _x: "wall",
        parameters={"eqa_vl": {"eqa_max_history": 2}},
    )
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["wall"])
    mem._relevant_objects = ["wall"]
    for _ in range(5):
        mem.query_answer("Is the wall blue? A) Yes B) No")
    iteration_lines = [c for c in captured["cmds"] if isinstance(c, str) and c.startswith("Iteration_")]
    assert len(iteration_lines) == 2
    # Should be the latest two iterations (history has 4 entries before the 5th call).
    assert iteration_lines[0].startswith("Iteration_2") and iteration_lines[1].startswith("Iteration_3")


def test_query_answer_memory_summary_gated_by_flag():
    """CONFIRMED_MEMORY is only sent to the VLM when memory_summary_enabled (Dynagraph)."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    captured = {"cmds": None}

    def fake_eqa(cmds):
        captured["cmds"] = cmds
        return "reasoning: r\nanswer: B\nconfidence: false\naction: none\nconfidence_reasoning: x"

    mem = GraphEQAMemory(eqa_client=fake_eqa, image_description_client=lambda _x: "sofa")
    mem.add_observation(rgb, np.array([4.0, -2.5, 0.5]), ["sofa"])
    mem._relevant_objects = ["sofa"]

    # Baseline GraphEQA: flag off -> no CONFIRMED_MEMORY block.
    mem.query_answer("Where is the sofa? A) x B) y")
    assert not any(isinstance(c, str) and "CONFIRMED_MEMORY" in c for c in captured["cmds"])

    # Dynagraph: flag on -> CONFIRMED_MEMORY block is included.
    mem.memory_summary_enabled = True
    mem._relevant_objects = ["sofa"]
    mem.query_answer("Where is the sofa? A) x B) y")
    assert any(isinstance(c, str) and "CONFIRMED_MEMORY" in c for c in captured["cmds"])


def test_query_answer_records_pregate_confidence_when_gated():
    """last_eqa_model_confident reflects the model's confidence before the coverage gate."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    raw = "reasoning: r\nanswer: B\nconfidence: true\naction: none\nconfidence_reasoning: sure"
    mem = GraphEQAMemory(eqa_client=lambda _c: raw, image_description_client=lambda _x: "wall")
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["wall"])
    # Relevant object never appears as a node label -> coverage gate suppresses confidence.
    mem._relevant_objects = ["unicorn"]
    _r, _a, gated_conf, _cr, _t, _imgs = mem.query_answer("Is there a unicorn? A) Yes B) No")
    assert gated_conf is False
    assert mem.last_eqa_model_confident is True


def test_query_answer_salvages_letter_on_caption_runaway():
    """When the main output never emits answer:, a terse retry recovers the letter."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    calls = {"n": 0}

    def fake_eqa(cmds):
        calls["n"] += 1
        if calls["n"] == 1:
            # Runaway: captions only, no answer field.
            return "caption:\n" + "\n".join(f"Image {i} shows a wall." for i in range(1, 40))
        return "B"

    mem = GraphEQAMemory(
        eqa_client=fake_eqa,
        image_description_client=lambda _x: "wall",
    )
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["wall"])
    mem._relevant_objects = ["wall"]
    _r, _a, _c, _cr, _t, _imgs = mem.query_answer("Is the wall blue? A) Yes B) No")
    assert calls["n"] == 2
    assert "answer:\nb" in mem.last_eqa_raw.lower()


def test_select_relevant_obs_ids_diversifies_views():
    """P2: selection reserves slots for keyword, frontier, and recent/spread views."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    # Three near-duplicate lamp views, plus a distant chair, plus a recent table.
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["lamp"])
    mem.add_observation(rgb, np.array([0.1, 0.0, 0.5]), ["lamp"])
    mem.add_observation(rgb, np.array([0.2, 0.0, 0.5]), ["lamp"])
    mem.add_observation(rgb, np.array([9.0, 9.0, 0.5]), ["chair"])
    mem.add_observation(rgb, np.array([5.0, -5.0, 0.5]), ["table"])
    mem._relevant_objects = ["lamp"]

    obs_ids = mem._select_relevant_obs_ids(max_images=4)
    assert len(obs_ids) == 4
    assert len(set(obs_ids)) == 4
    # At least one lamp view is included (keyword match).
    lamp_ids = {1, 2, 3}
    assert lamp_ids & set(obs_ids)
    # Not monopolized by the three duplicate lamp views: spread brings in others.
    assert not lamp_ids.issuperset(set(obs_ids))


def test_select_relevant_obs_ids_spatial_spread_after_frontier_sort():
    """Frontier placeholders are excluded from answer images; others still diversify."""
    from emet.memory.graph_eqa.graph_memory import replace

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["lamp"])
    mem.add_observation(rgb, np.array([4.0, 0.0, 0.5]), ["table"])
    mem.add_observation(rgb, np.array([8.0, 0.0, 0.5]), ["chair"])
    mem._relevant_objects = ["lamp"]
    # Mark obs 2 as frontier — must never be attached for answering.
    nodes = mem.get_nodes()
    for idx, node in enumerate(nodes):
        if int(node.obs_id) == 2:
            mem._nodes[idx] = replace(node, is_frontier=True)
            break

    obs_ids = mem._select_relevant_obs_ids(max_images=3)
    assert 2 not in obs_ids
    assert len(set(obs_ids)) == len(obs_ids)
    assert set(obs_ids) <= {1, 3}


def test_select_relevant_obs_ids_never_returns_frontier_placeholders():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["frontier"])
    mem._nodes[0].is_frontier = True
    mem.add_observation(rgb, np.array([1.0, 1.0, 0.5]), ["sofa"])
    mem._relevant_objects = ["clock"]
    obs_ids = mem._select_relevant_obs_ids(max_images=3)
    assert obs_ids == [2]
    assert all(not mem._obs_is_frontier(oid) for oid in obs_ids)


def test_select_relevant_obs_ids_uses_siglip_obs_grounder():
    """Dynagraph: a registered SigLIP obs grounder forces the target view in, even when the
    observation was captioned as something else (no keyword match)."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["couch"])
    mem.add_observation(rgb, np.array([1.0, 0.0, 0.5]), ["rug"])
    # obs 3 is the bed, but captioned "decorative object" -> no keyword hit for "bed".
    mem.add_observation(rgb, np.array([1.2, 0.0, 0.5]), ["decorative object"])
    mem.add_observation(rgb, np.array([1.4, 0.0, 0.5]), ["lamp"])
    mem.add_observation(rgb, np.array([1.6, 0.0, 0.5]), ["chair"])
    mem._relevant_objects = ["bed"]

    # No grounder: caption-keyword selection never surfaces the (mislabeled) bed first.
    assert mem._select_relevant_obs_ids(max_images=2)[0] != 3

    # With a SigLIP grounder mapping "bed" -> obs 3, the target view is surfaced first.
    mem.set_obs_id_grounder(lambda text: 3 if "bed" in text else None)
    obs_ids = mem._select_relevant_obs_ids(max_images=3)
    assert obs_ids[0] == 3


def test_select_relevant_obs_ids_no_keywords_uses_recent():
    """Without keyword objects, fall back to the most recent observations."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    for i in range(5):
        mem.add_observation(rgb, np.array([float(i), 0.0, 0.5]), [f"obj{i}"])
    mem._relevant_objects = None
    obs_ids = mem._select_relevant_obs_ids(max_images=2)
    assert obs_ids == [4, 5]


def test_display_image_index_maps_to_selected_obs_ids():
    """EQA action Image N must resolve through obs_ids order, not full observation list."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(
        eqa_client=lambda _cmds: (
            "reasoning: need lamp view\n"
            "answer: b\n"
            "confidence: false\n"
            "action: 2\n"
            "confidence_reasoning: check second attached image\n"
        ),
        image_description_client=lambda _x: "lamp",
    )
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["table"])
    mem.add_observation(rgb, np.array([1.0, 0.0, 0.5]), ["chair"])
    mem.add_observation(rgb, np.array([9.0, 9.0, 0.5]), ["lamp"])
    mem._relevant_objects = ["lamp"]
    obs_ids = mem._select_relevant_obs_ids(max_images=3)
    assert obs_ids[0] == 3
    assert obs_ids[1] == 1

    _r, _a, conf, _cr, target, _imgs = mem.query_answer("Where is the lamp?")
    assert conf is False
    assert target is not None
    # Image 2 -> obs_ids[1] == obs 1 (table at 0,0), not observations[1] (chair at 1,0).
    assert float(target[0]) == 0.0
    assert float(target[1]) == 0.0
    assert mem.last_eqa_obs_ids == obs_ids


def test_action_image_ref_accepts_graph_obs_id():
    """SCENE_GRAPH ``[Image 19]`` must resolve even when prompt only has Images 1..K."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["sofa"])
    # Simulate a frontier (or distant) observation with a large obs_id like HM-EQA.
    mem._observations.append(
        GraphObservation(
            obs_id=19,
            rgb=rgb,
            xyz=np.array([5.0, 5.0, 0.5]),
            labels=["frontier"],
            description="unexplored",
        )
    )
    assert mem._resolve_eqa_action_image_ref(1, [1]) == 1
    assert mem._resolve_eqa_action_image_ref(19, [1]) == 19
    assert mem._resolve_eqa_action_image_ref(99, [1]) is None


def test_query_answer_never_attaches_frontier_placeholder_rgb():
    """Holdout q104/q105 failfix4: answering off black 8×8 frontiers must not happen."""
    from emet.memory.graph_eqa.graph_memory import GraphNavigationSample, GraphNode, replace

    real_rgb = np.full((40, 40, 3), 120, dtype=np.uint8)
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    captured: dict = {"n_imgs": 0, "shapes": []}

    def _eqa(cmds):
        imgs = [c for c in cmds if hasattr(c, "size")]
        captured["n_imgs"] = len(imgs)
        captured["shapes"] = [tuple(np.asarray(im).shape) for im in imgs]
        return (
            "reasoning: need real view\n"
            "answer: unknown\n"
            "confidence: false\n"
            "action: 19\n"
            "confidence_reasoning: explore graph image 19\n"
        )

    mem = GraphEQAMemory(eqa_client=_eqa, image_description_client=lambda _x: "sofa")
    mem.add_observation(black, np.array([0.0, 0.0, 0.5]), ["frontier"])
    mem._nodes[0] = replace(mem._nodes[0], is_frontier=True)
    # Frontier-only graph would previously attach the placeholder; nav samples must win.
    mem._nav_samples.append(
        GraphNavigationSample(rgb=real_rgb, xyz=np.array([1.0, 2.0, 0.5]), base_xyz=np.array([0.0, 0.0, 0.0]))
    )
    mem._observations.append(
        GraphObservation(
            obs_id=19,
            rgb=black,
            xyz=np.array([5.0, 5.0, 0.5]),
            labels=["frontier"],
            description="unexplored",
        )
    )
    mem._nodes.append(
        GraphNode(
            node_id=99,
            labels=["frontier"],
            xyz=np.array([5.0, 5.0, 0.5]),
            obs_id=19,
            is_frontier=True,
        )
    )
    _r, _a, conf, _cr, _tp, imgs = mem.query_answer(
        "I'm looking for the fruit bowl. A) On the kitchen island B) On the dining table "
        "C) On the coffee table D) In the sunroom. Answer:"
    )
    assert conf is False
    assert mem.last_eqa_obs_ids == []
    assert mem.last_eqa_nav_fallback_count == 1
    assert captured["n_imgs"] == 1
    assert captured["shapes"] == [(40, 40, 3)]
    assert len(imgs) == 1
    assert tuple(np.asarray(imgs[0]).shape) == (40, 40, 3)
    # Action:19 is a graph obs_id, not prompt-local Image 19.
    assert mem.last_eqa_action_obs_id == 19


def test_navigation_waypoint_prefers_viewpoint_when_far():
    """Image-N nav should target the capture viewpoint when the robot is elsewhere."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    viewer = np.array([1.0, 2.0, 0.0], dtype=np.float64)
    mem.add_observation(rgb, np.array([5.0, 6.0, 0.5]), ["lamp"], viewer_xyz=viewer)
    pt = mem._navigation_waypoint_for_obs(1, np.array([9.0, 9.0, 0.0]))
    assert pt is not None
    assert float(pt[0]) == 1.0
    assert float(pt[1]) == 2.0


def test_navigation_waypoint_standoff_when_at_viewpoint():
    """When already at the capture pose, advance toward the object anchor instead of obs.xyz."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    viewer = np.array([1.0, 2.0, 0.0], dtype=np.float64)
    mem.add_observation(rgb, np.array([1.35, 2.0, 0.5]), ["lamp"], viewer_xyz=viewer)
    pt = mem._navigation_waypoint_for_obs(1, np.array([1.0, 2.0, 0.0]))
    assert pt is not None
    assert float(pt[0]) > 1.0
    assert abs(float(pt[1]) - 2.0) < 1e-6


def test_navigation_waypoint_keeps_frontier_anchor():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem._nodes.append(
        GraphNode(
            node_id=1,
            labels=["frontier"],
            xyz=np.array([3.0, 4.0, 0.0]),
            obs_id=1,
            is_frontier=True,
        )
    )
    mem._observations.append(
        GraphObservation(
            obs_id=1,
            rgb=rgb,
            xyz=np.array([3.0, 4.0, 0.0]),
            labels=["frontier"],
            description="frontier",
        )
    )
    pt = mem._navigation_waypoint_for_obs(1, np.array([0.0, 0.0, 0.0]))
    assert float(pt[0]) == 3.0
    assert float(pt[1]) == 4.0


def test_near_heuristic():
    """_near returns True when 2D distance <= max_dist."""
    assert _near(np.array([0, 0, 0]), np.array([0.5, 0, 0]), max_dist=1.0) is True
    assert _near(np.array([0, 0, 0]), np.array([2, 0, 0]), max_dist=1.0) is False


def test_add_observation_stores_bbox_xyxy_on_node():
    """Instance graph nodes keep a pixel crop for Dynagraph Rerun thumbnails."""
    mem = GraphEQAMemory(defer_llm_clients=True)
    rgb = np.zeros((60, 80, 3), dtype=np.uint8)
    mem.add_observation(
        rgb,
        np.array([1.0, 2.0, 0.5]),
        ["mug"],
        bbox_xyxy=(10, 20, 40, 55),
    )
    node = mem.get_nodes()[0]
    assert node.bbox_xyxy == (10, 20, 40, 55)


def test_seen_from_edge_links_object_to_viewpoint_node():
    """``seen_from`` targets a viewpoint graph node at ``viewer_xyz``."""
    mem = GraphEQAMemory(defer_llm_clients=True)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    viewer = np.array([1.0, 2.0, 0.0], dtype=np.float64)
    mem.add_observation(
        rgb,
        np.array([3.0, 4.0, 0.9]),
        ["mug"],
        viewer_xyz=viewer,
    )
    nodes = mem.get_nodes()
    objects = [n for n in nodes if not n.is_viewpoint]
    viewpoints = [n for n in nodes if n.is_viewpoint]
    assert len(objects) == 1
    assert len(viewpoints) == 1
    np.testing.assert_allclose(viewpoints[0].xyz, viewer)
    edges = mem.get_edges()
    assert (objects[0].node_id, viewpoints[0].node_id, "seen_from") in edges
    obs = mem.get_observations()[0]
    assert obs.viewer_xyz is not None
    np.testing.assert_allclose(obs.viewer_xyz, viewer)
    s = mem.to_string()
    assert "seen_from" in s
    assert "View " in s


def test_on_floor_heuristic():
    """_on_floor returns True when z <= threshold."""
    assert _on_floor(np.array([0, 0, 0.02])) is True
    assert _on_floor(np.array([0, 0, 0.2])) is False


def test_query_answer_returns_tuple_with_mock_client():
    """query_answer returns (reasoning, answer, confidence, confidence_reasoning, target_point, relevant_images)."""

    def mock_eqa(commands):
        return "reasoning: I see a table.\nanswer: Yes\nconfidence: true\naction: \nconfidence_reasoning: Sure."

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


def test_color_question_answer_contains_red_and_blue():
    """
    Default MuJoCo scene has a red cylinder and blue cube; GraphEQA answer should name both colors.

    Uses mocks (no GPU LLM). Sim coverage: test_graph_eqa_color_question_default_mujoco_scene.
    """

    def mock_eqa(commands):
        return (
            "reasoning: the scene graph lists a red cylinder and a blue cube on the table.\n"
            "answer: red and blue (red cylinder, blue cube).\n"
            "confidence: true\n"
            "action: \n"
            "confidence_reasoning: both objects are in the graph labels.\n"
        )

    mem = GraphEQAMemory(
        eqa_client=mock_eqa,
        image_description_client=lambda cmd: "red cylinder, blue cube",
    )
    rgb = np.zeros((60, 80, 3), dtype=np.uint8)
    # Same layout as default scene: object2 / object1 positions (see test_red_cylinder_in_sim)
    mem.add_observation(rgb, np.array([0.08, -0.55, 0.6]), ["red cylinder"])
    mem.add_observation(rgb, np.array([-0.02, -0.55, 0.6]), ["blue cube"])
    graph_str = mem.to_string().lower()
    assert "red" in graph_str
    assert "blue" in graph_str

    _reasoning, answer, confidence, _cr, _tp, _imgs = mem.query_answer("Which color objects can you see?", None, None)
    assert confidence is True
    al = answer.lower()
    assert "red" in al, f"expected 'red' in answer, got: {answer!r}"
    assert "blue" in al, f"expected 'blue' in answer, got: {answer!r}"


def test_to_tree_string():
    """to_tree_string formats the scene graph as an indented tree with Floor and relations."""
    mem = GraphEQAMemory(
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    rgb = np.zeros((60, 80, 3), dtype=np.uint8)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.0]), ["carpet"])
    mem.add_observation(rgb, np.array([0.5, 0.0, 0.8]), ["table"])
    mem.add_observation(rgb, np.array([0.5, 0.0, 0.85]), ["cup"], description="A red cup on the table")
    tree = mem.to_tree_string()
    assert "Scene (3D spatial graph)" in tree
    assert "Floor" in tree
    assert "carpet" in tree
    assert "table" in tree
    assert "cup" in tree
    assert "0.50" in tree
    assert "A red cup" in tree or "red cup" in tree


def test_print_memory():
    """print_memory returns the same tree as to_tree_string."""
    mem = GraphEQAMemory(
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    mem.add_observation(
        np.zeros((60, 80, 3), dtype=np.uint8),
        np.array([0.0, 0.0, 0.1]),
        ["sofa"],
    )
    out = mem.print_memory()
    assert "Scene" in out
    assert "sofa" in out


def test_graph_eqa_backend_print_memory():
    """GraphEQABackend.print_memory delegates to graph and returns tree text."""
    mem = GraphEQAMemory(
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    mem.add_observation(
        np.zeros((60, 80, 3), dtype=np.uint8),
        np.array([0.0, 0.0, 0.0]),
        ["lamp"],
    )
    backend = GraphEQABackend(mem)
    text = backend.print_memory()
    assert "Scene" in text
    assert "lamp" in text


def test_graph_eqa_save_load_roundtrip():
    """Save and load graph memory restores nodes, edges, observations (xyz and labels)."""
    mem = GraphEQAMemory(
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    mem.add_observation(rgb, np.array([1.0, 2.0, 0.5]), ["chair"], description="A wooden chair")
    mem.add_observation(rgb, np.array([1.1, 2.1, 0.5]), ["desk"])
    backend = GraphEQABackend(mem)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "graph_mem"
        backend.save(str(path))
        assert (path / "manifest.json").exists()
        assert (path / "graph.json").exists()
        mem2 = GraphEQAMemory(eqa_client=lambda x: "", image_description_client=lambda x: "")
        backend2 = GraphEQABackend(mem2)
        backend2.load(str(path))
        nodes = mem2.get_nodes()
        obs = mem2.get_observations()
        assert len(nodes) == 2
        assert len(obs) == 2
        n1 = next(n for n in nodes if n.node_id == 1)
        assert "chair" in n1.labels
        assert n1.description == "A wooden chair"
        o1 = next(o for o in obs if o.obs_id == 1)
        assert list(o1.xyz) == [1.0, 2.0, 0.5]
        assert o1.labels == ["chair"]
        assert o1.description == "A wooden chair"


def test_labels_are_semantic_graph_hypothesis():
    assert labels_are_semantic_graph_hypothesis(["cup"]) is True
    assert labels_are_semantic_graph_hypothesis(["object"]) is False
    assert labels_are_semantic_graph_hypothesis(["OBJECT"]) is False
    assert labels_are_semantic_graph_hypothesis(["object", "mug"]) is True
    assert labels_are_semantic_graph_hypothesis([]) is False


def test_record_navigation_sample_adds_viewpoint_not_object_node():
    mem = GraphEQAMemory(eqa_client=lambda x: "", image_description_client=lambda x: "")
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    base = np.array([1.1, 2.0, 0.0])
    mem.record_navigation_sample(rgb, np.array([1.0, 2.0, 0.1]), base_xyz=base)
    nodes = mem.get_nodes()
    assert len(mem.get_observations()) == 0
    assert len(mem.get_navigation_samples()) == 1
    assert len(nodes) == 1
    assert nodes[0].is_viewpoint
    np.testing.assert_allclose(nodes[0].xyz, base)


def test_viewpoint_spatial_merge_reuses_node():
    mem = GraphEQAMemory(
        parameters={"dynagraph_viewpoint_merge_m": 0.25},
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    base_a = np.array([1.0, 2.0, 0.0])
    base_b = np.array([1.05, 2.02, 0.01])
    mem.record_navigation_sample(rgb, np.array([1.0, 2.0, 0.1]), base_xyz=base_a)
    mem.record_navigation_sample(rgb, np.array([1.0, 2.0, 0.1]), base_xyz=base_b)
    viewpoints = [n for n in mem.get_nodes() if n.is_viewpoint]
    assert len(viewpoints) == 1
    np.testing.assert_allclose(viewpoints[0].xyz, base_b)


def test_record_navigation_respects_graph_eqa_record_navigation_false():
    mem = GraphEQAMemory(
        parameters={"graph_eqa_record_navigation": False},
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    mem.record_navigation_sample(rgb, np.array([0.0, 0.0, 0.0]))
    assert mem.get_navigation_samples() == []


def test_query_answer_navigation_fallback_images_and_target():
    mem = GraphEQAMemory(
        eqa_client=lambda x: "reasoning: r\nanswer: no\nconfidence: false\naction: 1\nconfidence_reasoning: look",
        image_description_client=lambda q: "mug",
    )
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    mem.record_navigation_sample(rgb, np.array([0.5, -0.25, 0.12]), base_xyz=np.array([0.0, 0.0, 0.0]))
    _r, _a, _c, _cr, target_point, imgs = mem.query_answer("Is there a mug?", None, None)
    assert len(imgs) == 1
    assert target_point is not None
    assert abs(float(target_point[0]) - 0.5) < 1e-5
    assert abs(float(target_point[1]) - (-0.25)) < 1e-5


def test_dynagraph_spatial_merge_keeps_detection_bbox():
    mem = GraphEQAMemory(
        parameters={"dynagraph_merge_xy_m": 1.0},
        defer_llm_clients=True,
    )
    rgb = np.zeros((80, 60, 3), dtype=np.uint8)
    mem.set_graph_timestep(1)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["cup"], bbox_xyxy=(5, 5, 25, 30))
    mem.set_graph_timestep(2)
    mem.add_observation(rgb, np.array([0.15, 0.0, 0.5]), ["cup"], bbox_xyxy=(30, 10, 50, 35))
    node = mem.get_nodes()[0]
    assert not node.is_viewpoint
    assert node.bbox_xyxy == (30, 10, 50, 35)


def test_dynagraph_spatial_merge_same_obs_id():
    mem = GraphEQAMemory(
        parameters={"dynagraph_merge_xy_m": 1.0},
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem.set_graph_timestep(1)
    id1 = mem.add_observation(rgb, np.array([0.0, 0.0, 0.5]), ["cup"])
    mem.set_graph_timestep(2)
    id2 = mem.add_observation(rgb, np.array([0.2, 0.0, 0.5]), ["cup"])
    assert id1 == id2
    assert len(mem.get_nodes()) == 1
    assert mem.get_nodes()[0].support_count == 2


def test_heuristic_relevant_objects_from_question():
    from emet.memory.graph_eqa.graph_memory import heuristic_relevant_objects

    objs = heuristic_relevant_objects("Is the lamp next to the bed on?")
    assert "lamp" in objs
    assert "bed" in objs


def test_graph_covers_relevant_objects_requires_all_keywords():
    mem = GraphEQAMemory(eqa_client=lambda x: "", image_description_client=lambda x: "")
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.1]), ["lamp"])
    mem._relevant_objects = ["lamp", "bed"]
    assert not mem._graph_covers_relevant_objects()
    mem.add_observation(rgb, np.array([1.0, 0.0, 0.1]), ["bed"])
    assert mem._graph_covers_relevant_objects()


def test_dynagraph_maintain_prunes_stale_nodes():
    mem = GraphEQAMemory(
        parameters={"dynagraph_staleness_horizon": 5},
        eqa_client=lambda x: "",
        image_description_client=lambda x: "",
    )
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    mem.set_graph_timestep(1)
    mem.add_observation(rgb, np.array([0.0, 0.0, 0.1]), ["table"])
    mem.set_graph_timestep(16)
    mem.add_observation(rgb, np.array([2.0, 0.0, 0.1]), ["chair"])
    removed = mem.maintain(current_step=20)
    assert removed == 1
    assert len(mem.get_nodes()) == 1
    assert mem.get_nodes()[0].node_id == 1
    assert "chair" in mem.get_nodes()[0].labels


def test_label_matches_relevant_object_phrase_vs_short_label():
    assert label_matches_relevant_object("standing fan", "fan")
    assert label_matches_relevant_object("fan", "standing fan")
    assert label_matches_relevant_object("woven basket", "basket")
    assert not label_matches_relevant_object("television", "fan")
    assert label_matches_relevant_object("armchair", "chair")


def test_record_nav_attempt_updates_node_metadata():
    from emet.memory.graph_eqa.graph_memory import GraphEQAMemory, GraphNode

    mem = GraphEQAMemory(defer_llm_clients=True)
    mem._nodes = [
        GraphNode(
            node_id=1,
            obs_id=3,
            xyz=np.array([1.0, 2.0, 0.0]),
            labels=["frontier"],
            is_frontier=True,
        )
    ]
    mem.record_nav_attempt(3, success=False, note="navmesh_no_path", dist_m=0.0, step=7)
    node = mem.get_nodes()[0]
    assert node.nav_attempts == 1
    assert node.nav_failures == 1
    assert node.last_nav_note == "navmesh_no_path"
    assert node.last_nav_at_step == 7
    assert "unreachable" in mem.to_string()


def test_alternate_nav_target_skips_failed_frontier_obs():
    from emet.memory.graph_eqa.graph_memory import GraphEQAMemory, GraphNode

    mem = GraphEQAMemory(defer_llm_clients=True)
    mem._nodes = [
        GraphNode(
            node_id=1,
            obs_id=3,
            xyz=np.array([1.0, 2.0, 0.0]),
            labels=["frontier"],
            is_frontier=True,
            nav_failures=2,
        ),
        GraphNode(
            node_id=2,
            obs_id=5,
            xyz=np.array([4.0, 5.0, 0.0]),
            labels=["frontier"],
            is_frontier=True,
            nav_failures=0,
        ),
    ]
    alt = mem.alternate_nav_target_for_failed_action("where is the basket?", 3, None, None)
    assert alt is not None
    assert abs(float(alt[0]) - 4.0) < 1e-6
    assert abs(float(alt[1]) - 5.0) < 1e-6

