# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from pathlib import Path

import pytest

from emet.benchmarks.sqa3d.datasets import get_sqa3d_question, load_sqa3d_questions
from emet.benchmarks.sqa3d.prompts import format_sqa3d_prompt

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_load_sqa3d_fixture_questions():
    qs = load_sqa3d_questions(
        "val",
        questions_path=FIXTURES / "v1_balanced_questions_val_scannetv2.json",
        annotations_path=FIXTURES / "v1_balanced_sqa_annotations_val_scannetv2.json",
    )
    assert len(qs) == 3
    assert qs[0].question_id == 220602000000
    assert qs[0].scene_id == "scene0380_00"
    assert qs[0].primary_answer == "brown"
    assert qs[0].position[0] == pytest.approx(-0.9651003385573296)
    assert len(qs[0].alternative_situations) == 2


def test_get_sqa3d_question_by_id():
    qs = load_sqa3d_questions(
        "val",
        questions_path=FIXTURES / "v1_balanced_questions_val_scannetv2.json",
        annotations_path=FIXTURES / "v1_balanced_sqa_annotations_val_scannetv2.json",
    )
    q = get_sqa3d_question(qs, question_id=220602000001)
    assert q.question.startswith("Is ")


def test_format_sqa3d_prompt():
    text = format_sqa3d_prompt("I face a window.", "What color is the desk?")
    assert "Situation:" in text
    assert "Question:" in text


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_sqa3d_questions("val", questions_path=FIXTURES / "missing.json")
