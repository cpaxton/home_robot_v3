# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from emet.habitat.metrics import EpisodeMetrics, extract_mcq_letter, grade_mcq_answer, summarize_episodes


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
