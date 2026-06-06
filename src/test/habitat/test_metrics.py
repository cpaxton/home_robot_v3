# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from emet.habitat.metrics import (
    EpisodeMetrics,
    compare_method_results,
    extract_mcq_letter,
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


def test_read_completed_question_ids(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text(
        '{"question_id": 0, "correct": true}\n{"question_id": 2, "correct": false}\n',
        encoding="utf-8",
    )
    assert read_completed_question_ids(path) == {0, 2}
    assert read_completed_question_ids(tmp_path / "missing.jsonl") == set()
