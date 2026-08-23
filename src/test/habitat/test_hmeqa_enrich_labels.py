# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from emet.habitat.hmeqa_enrich_labels import (
    GRAPHEQA_HMEQA_QUESTION_COUNT,
    HMEQA_PAPER_QUESTION_COUNT,
    enrich_labels_for_dataset_question,
    enrich_labels_for_question,
    grapheqa_baseline_question_ids,
    hmeqa_paper_question_ids,
    load_hmeqa_enrich_labels,
    parse_enrich_label_text,
)


def test_hmeqa_paper_question_ids():
    ids = hmeqa_paper_question_ids()
    assert len(ids) == HMEQA_PAPER_QUESTION_COUNT
    assert ids[0] == 0
    assert ids[-1] == HMEQA_PAPER_QUESTION_COUNT - 1


def test_parse_enrich_label_text_skips_unknown():
    assert parse_enrich_label_text("kettle. unknown.") == ["kettle"]
    assert "books" in parse_enrich_label_text("books. window. dining table.")


def test_bundled_enrich_labels_cover_paper_questions():
    table = load_hmeqa_enrich_labels()
    assert len(table) == GRAPHEQA_HMEQA_QUESTION_COUNT
    assert enrich_labels_for_question(0, "00006-HkseAnWCgqk") == "blanket."


def test_grapheqa_baseline_question_ids_match_enrich_episodes(tmp_path):
    """The mapping follows the exact upstream filtered sequence, not scene count alone."""
    questions = tmp_path / "questions.csv"
    questions.write_text(
        "scene,floor,question,choices,question_formatted,answer,label\n"
        "outside,0,q,\"['a','b','c','d']\",q,A,state\n"
        "scene-a,0,q,\"['a','b','c','d']\",q,A,state\n"
        "scene-b,0,q,\"['a','b','c','d']\",q,A,state\n"
        "scene-b,1,q,\"['a','b','c','d']\",q,A,state\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "0_scene-a:\n  labels: alpha.\n1_scene-b:\n  labels: beta.\n2_scene-b:\n  labels: gamma.\n",
        encoding="utf-8",
    )

    ids = grapheqa_baseline_question_ids(
        questions_path=questions,
        labels_path=labels,
    )
    assert ids == [1, 2, 3]
    assert (
        enrich_labels_for_dataset_question(
            2,
            "scene-b",
            questions_path=questions,
            labels_path=labels,
        )
        == "beta."
    )
    assert (
        enrich_labels_for_dataset_question(
            0,
            "outside",
            questions_path=questions,
            labels_path=labels,
        )
        == ""
    )
