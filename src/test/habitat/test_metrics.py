# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from emet.habitat.metrics import (
    EpisodeMetrics,
    choices_are_attribute_state,
    choices_are_location_mcq,
    compare_method_results,
    episode_run_completed,
    extract_mcq_letter,
    extract_mcq_letter_from_raw_eqa,
    grade_mcq_answer,
    parse_mcq_choices_from_question,
    question_is_visibility_location,
    read_completed_question_ids,
    should_abstain_location_mcq,
    summarize_episodes,
)


def test_grade_mcq_answer_letter():
    assert grade_mcq_answer("B", "B")
    assert grade_mcq_answer("Answer: B", "B")
    assert not grade_mcq_answer("C", "B")
    assert extract_mcq_letter("Answer: c") == "C"
    assert grade_mcq_answer("The lamp is off", "B", choices=["on", "off", "none", "broken"])


def test_on_the_place_choices_are_location_not_attribute():
    """Holdout q105: 'On the kitchen island' must not count as on/off attribute."""
    place = [
        "On the kitchen island",
        "On the dining table",
        "On the coffee table",
        "In the sunroom",
    ]
    assert choices_are_location_mcq(place)
    assert not choices_are_attribute_state(place)
    assert choices_are_attribute_state(["On", "Off", "Unknown", "(Do not choose)"])
    assert choices_are_attribute_state(["turned on", "turned off", "open", "closed"])


def test_q65_ac_on_off_is_attribute_not_location():
    """Holdout q65: leave-AC-on? with 'it is off/on' choices must not location-salvage."""
    from emet.habitat.metrics import question_is_attribute_state

    q = (
        "Did I leave the air conditioning in the living room on? "
        "A) No, it is off B) (Do not choose this option) "
        "C) Yes, it is on D) (Do not choose this option). Answer:"
    )
    choices = parse_mcq_choices_from_question(q)
    assert question_is_attribute_state(q)
    assert choices_are_attribute_state(choices)
    assert not choices_are_location_mcq(choices)


def test_count_choices_are_not_location_mcq():
    from emet.habitat.metrics import choices_are_count_mcq

    counts = ["Three", "One", "None", "Two"]
    assert choices_are_count_mcq(counts)
    assert not choices_are_location_mcq(counts)


def test_grade_mcq_answer_letter_e():
    assert extract_mcq_letter("E") == "E"
    assert extract_mcq_letter("Answer: e") == "E"
    assert grade_mcq_answer("E", "E")
    assert grade_mcq_answer("Answer: E", "E")
    assert not grade_mcq_answer("D", "E")
    choices = ["a", "b", "c", "d", "none of the above"]
    assert extract_mcq_letter("none of the above", choices) == "E"
    assert extract_mcq_letter_from_raw_eqa("Answer: e\nConfidence: true", choices) == "E"


def test_extract_mcq_letter_from_raw_eqa_prefers_answer_field():
    raw = (
        "Caption:\nImage 1 shows a room.\n"
        "Reasoning:\nNo grill visible yet.\n"
        "Answer:\nNo\n"
        "Confidence:\nFALSE\n"
        "Action:\n1\n"
        "Confidence_reasoning:\nExplore more.\n"
    )
    choices = ["Yes", "No", "Partially", "Cannot tell"]
    assert extract_mcq_letter_from_raw_eqa(raw, choices) == "B"
    assert extract_mcq_letter_from_raw_eqa("Answer: b\nConfidence: true", choices) == "B"


def test_extract_mcq_letter_choice_word_boundaries():
    choices = ["on", "off", "none", "broken"]
    assert extract_mcq_letter("unknown", choices) == ""
    assert extract_mcq_letter("off", choices) == "B"


def test_extract_mcq_letter_does_not_match_article_a_in_prose():
    choices = ["(Do not choose this option)", "Yes, it is on", "No, it is not on", "(Do not choose this option)"]
    caption = "caption:\nImage 1 is a view of video player controls. Image 2 is a frontier"
    assert extract_mcq_letter(caption, choices) == ""
    assert extract_mcq_letter_from_raw_eqa(caption, choices) == ""


def test_extract_mcq_letter_from_raw_eqa_blank_answer_field():
    raw = "reasoning:\nNeed more exploration.\nanswer:\nconfidence:\nFALSE\naction:\n3\n"
    assert extract_mcq_letter_from_raw_eqa(raw) == ""
    assert not grade_mcq_answer("", "B")


def test_location_mcq_abstains_on_visibility_no():
    """Q17-style: location choices but model answers visibility ``No`` → no letter."""
    raw = (
        "Reasoning:\nI do not see a woven basket.\n"
        "Answer:\nNo\n"
        "Confidence:\nTRUE\n"
    )
    choices = [
        "In the bedroom",
        "On the table",
        "Next to the sofa",
        "Next to the armchair",
    ]
    assert should_abstain_location_mcq(raw, choices)
    assert extract_mcq_letter_from_raw_eqa(raw, choices) == ""


def test_location_mcq_salvage_answer_overrides_visibility_no():
    raw = (
        "Answer:\nNo\nConfidence:\nTRUE\n"
        "[salvage-location]\nanswer:\nD\n"
    )
    choices = [
        "By the kitchen counter",
        "Between TV and living room sofas",
        "Next to the dining table",
        "Next to the living room armchairs",
    ]
    assert not should_abstain_location_mcq(raw, choices)
    assert extract_mcq_letter_from_raw_eqa(raw, choices) == "D"


def test_parse_mcq_choices_from_question_formatted():
    q = (
        "Did you see the woven basket anywhere? "
        "A) By the kitchen counter B) Between TV and living room sofas "
        "C) Next to the dining table D) Next to the living room armchairs. Answer:"
    )
    assert parse_mcq_choices_from_question(q) == [
        "By the kitchen counter",
        "Between TV and living room sofas",
        "Next to the dining table",
        "Next to the living room armchairs",
    ]
    assert question_is_visibility_location(q)


def test_yes_no_mcq_still_maps_no_to_b():
    raw = "Answer:\nNo\nConfidence:\nTRUE\n"
    choices = ["Yes", "No", "Partially", "Cannot tell"]
    assert not should_abstain_location_mcq(raw, choices)
    assert extract_mcq_letter_from_raw_eqa(raw, choices) == "B"


def test_extract_mcq_letter_ignores_answer_in_prose():
    raw = "reasoning:\nI cannot answer the question yet.\nAnswer:\nB\nConfidence:\nTRUE\n"
    assert extract_mcq_letter_from_raw_eqa(raw) == "B"


def test_summarize_episodes():
    eps = [
        EpisodeMetrics(
            dataset="hmeqa",
            method="dynagraph",
            question_id=0,
            scene="s",
            floor=0,
            question="q",
            gold_answer_letter="A",
            predicted_answer="A",
            correct=True,
            confident=True,
            planning_steps=5,
            success=True,
        ),
        EpisodeMetrics(
            dataset="hmeqa",
            method="dynagraph",
            question_id=1,
            scene="s2",
            floor=0,
            question="q2",
            gold_answer_letter="B",
            predicted_answer="C",
            correct=False,
            confident=False,
            planning_steps=10,
            success=False,
        ),
    ]
    s = summarize_episodes(eps)
    assert s["n"] == 2.0
    assert s["accuracy"] == 0.5
    assert s["mean_steps"] == 7.5


def test_compare_method_results():
    graph = [
        EpisodeMetrics(
            dataset="hmeqa",
            method="static_graph",
            question_id=0,
            scene="s",
            floor=0,
            question="q",
            gold_answer_letter="B",
            predicted_answer="B",
            correct=True,
            confident=True,
            planning_steps=10,
            success=True,
            parsed_answer_letter="B",
        ),
    ]
    dyna = [
        EpisodeMetrics(
            dataset="hmeqa",
            method="dynagraph",
            question_id=0,
            scene="s",
            floor=0,
            question="q",
            gold_answer_letter="B",
            predicted_answer="C",
            correct=False,
            confident=True,
            planning_steps=12,
            success=False,
            parsed_answer_letter="C",
        ),
    ]
    cmp = compare_method_results(graph, dyna)
    assert cmp["static_graph"]["accuracy"] == 1.0
    assert cmp["dynagraph"]["accuracy"] == 0.0
    assert cmp["static_only"] == 1
    assert cmp["dynagraph_only"] == 0
    assert cmp["per_question"][0]["question_id"] == 0


def test_episode_run_completed():
    assert episode_run_completed({"planning_steps": 5, "eqa_iterations": 3})
    assert episode_run_completed({"planning_steps": 56, "raw_eqa_output": "Answer:\nB"})
    assert not episode_run_completed({"planning_steps": 56, "eqa_iterations": 0, "raw_eqa_output": ""})
    assert not episode_run_completed({"planning_steps": 0, "error": "CUDA OOM"})
    assert episode_run_completed({"correct": True})  # legacy row without planning_steps


def test_read_completed_question_ids(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text(
        '{"question_id": 0, "correct": true, "planning_steps": 5, "eqa_iterations": 2}\n'
        '{"question_id": 1, "correct": false, "planning_steps": 56, "eqa_iterations": 0}\n'
        '{"question_id": 2, "correct": false, "error": "CUDA OOM"}\n'
        '{"question_id": 3, "correct": false, "planning_steps": 0}\n',
        encoding="utf-8",
    )
    assert read_completed_question_ids(path) == {0}
    assert read_completed_question_ids(tmp_path / "missing.jsonl") == set()
