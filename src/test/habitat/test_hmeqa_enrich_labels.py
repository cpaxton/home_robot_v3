# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from emet.habitat.hmeqa_enrich_labels import (
    HMEQA_PAPER_QUESTION_COUNT,
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
    assert len(table) >= HMEQA_PAPER_QUESTION_COUNT
    assert enrich_labels_for_question(0, "00006-HkseAnWCgqk") == "blanket."


def test_grapheqa_baseline_question_ids_match_enrich_episodes(tmp_path):
    """The 114 GraphEQA paper episodes map to CSV rows on the enrich scenes."""
    import csv

    from emet.habitat.datasets import load_hmeqa_questions

    ids = grapheqa_baseline_question_ids()
    assert len(ids) == 114
    questions = load_hmeqa_questions()
    table = load_hmeqa_enrich_labels()
    ge_scenes = {str(k).split("_", 1)[1] for k in table}
    # Every selected row is on an enrich scene, and enrich qid == position among them.
    ge_rows = [q for q in questions if q.scene in ge_scenes]
    assert len(ge_rows) == 114
    for eid, i in enumerate(ids):
        assert questions[i].scene == ge_rows[eid].scene
        key = f"{eid}_{questions[i].scene}"
        assert key in table
