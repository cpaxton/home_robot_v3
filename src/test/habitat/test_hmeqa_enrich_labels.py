# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from emet.habitat.hmeqa_enrich_labels import (
    HMEQA_PAPER_QUESTION_COUNT,
    enrich_labels_for_question,
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
