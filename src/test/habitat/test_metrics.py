# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from emet.habitat.metrics import (
    EpisodeMetrics,
    compare_method_results,
    episode_run_completed,
    extract_mcq_letter,
    extract_mcq_letter_from_raw_eqa,
    grade_mcq_answer,
    read_completed_question_ids,
    summarize_episodes,
)


def test_grade_mcq_answer_letter():
    assert grade_mcq_answer("B", "B")
    assert grade_mcq_answer("Answer: B", "B")
    assert not grade_mcq_answer("C", "B")
    assert extract_mcq_letter("Answer: c") == "C"
    assert grade_mcq_answer("The lamp is off", "B", choices=["on", "off", "none", "broken"])


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
            method="graph_eqa",
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
    assert cmp["graph_eqa"]["accuracy"] == 1.0
    assert cmp["dynagraph"]["accuracy"] == 0.0
    assert cmp["graph_only"] == 1
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
