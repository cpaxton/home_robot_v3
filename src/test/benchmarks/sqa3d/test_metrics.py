# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from pathlib import Path

from emet.benchmarks.sqa3d.datasets import load_sqa3d_questions
from emet.benchmarks.sqa3d.metrics import (
    answer_match,
    clean_answer,
    load_predictions,
    score_sqa3d_predictions,
    summarize_localization,
)
from emet.benchmarks.sqa3d.question_types import question_type_index

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _fixture_questions():
    return load_sqa3d_questions(
        "val",
        questions_path=FIXTURES / "v1_balanced_questions_val_scannetv2.json",
        annotations_path=FIXTURES / "v1_balanced_sqa_annotations_val_scannetv2.json",
    )


def test_clean_answer_digits():
    assert clean_answer("2") == "two"
    assert clean_answer("brown") == "brown"


def test_answer_match_exact_and_refined():
    assert answer_match("brown", ["brown"]) == (True, True)
    assert answer_match("brown desk", ["brown"]) == (False, True)
    assert answer_match("red", ["brown"]) == (False, False)


def test_question_type_index():
    assert question_type_index("What color is it?") == 0
    assert question_type_index("Is the lamp on?") == 1
    assert question_type_index("Where is it?") == 5


def test_score_sqa3d_predictions():
    qs = _fixture_questions()
    preds = {
        220602000000: "brown",
        220602000001: "yes",
        220602000002: "2",
    }
    out = score_sqa3d_predictions(qs, preds)
    assert out["em@1"] == 1.0
    assert out["n_scored"] == 3.0
    assert "what" in out["by_question_type"]


def test_load_predictions_jsonl(tmp_path: Path):
    path = tmp_path / "preds.jsonl"
    path.write_text(
        '{"question_id": 1, "answer": "brown"}\n{"question_id": 2, "text": "yes"}\n',
        encoding="utf-8",
    )
    preds = load_predictions(path)
    assert preds[1] == "brown"
    assert preds[2] == "yes"


def test_summarize_localization_perfect():
    gt_p = [(0.0, 0.0, 0.0)]
    gt_r = [(0.0, 0.0, 0.0, 1.0)]
    pred_p = [[(0.1, 0.1, 0.0)]]
    pred_r = [[(0.0, 0.0, 0.05, 0.999)]]
    m = summarize_localization(gt_p, gt_r, pred_p, pred_r)
    assert m["acc@0.5m"] == 1.0
    assert m["acc@1.0m"] == 1.0


def test_summarize_localization_random_miss():
    gt_p = [(0.0, 0.0, 0.0)]
    gt_r = [(0.0, 0.0, 0.0, 1.0)]
    pred_p = [[(5.0, 5.0, 0.0)]]
    pred_r = [[(0.0, 0.0, 0.0, 1.0)]]
    m = summarize_localization(gt_p, gt_r, pred_p, pred_r)
    assert m["acc@0.5m"] == 0.0
    assert m["acc@1.0m"] == 0.0
