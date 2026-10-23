# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import pytest

from emet.eval.embodied_qa_datasets import abstain_questions, development_pilot, express_questions


def abstain_row(qid="q1"):
    return {
        "question_id": qid,
        "question": "What is on the wall?",
        "answer": "unknown",
        "category": "Referential Underspecification",
        "episode_history": "scene",
        "frame_list": [1, 3],
    }


def test_express_keeps_id_but_withholds_evaluation_annotations():
    record = {
        "episode_id": 0,
        "question": "What color?",
        "answer": "blue",
        "type": "attribute",
        "scene_id": "hm3d/scene",
        "goal_position": [1, 2, 3],
        "actions": {"step_0": "oracle"},
    }
    question = express_questions([record])[0]
    assert question.question_id == "0" and question.reference_answer == "blue"
    assert question.model_input() == {"question": "What color?"}


def test_abstain_pair_identity_and_internal_pilot():
    rows = [abstain_row("q2"), abstain_row("q1")]
    questions = abstain_questions(rows, rows)
    assert {(q.question_id, q.variant) for q in development_pilot(questions, 1)} == {
        ("q1", "answerable"),
        ("q1", "unanswerable"),
    }
    assert questions[0].model_input() == {"question": "What is on the wall?"}


@pytest.mark.parametrize("failure", ["empty", "duplicate", "ids", "frames", "scene"])
def test_abstain_rejects_incomplete_or_misaligned_pairs(failure):
    positive, negative = [abstain_row()], [abstain_row()]
    if failure == "empty":
        positive = []
    elif failure == "duplicate":
        positive *= 2
    elif failure == "ids":
        negative[0]["question_id"] = "other"
    elif failure == "frames":
        negative[0]["frame_list"] = [2]
    else:
        negative[0]["episode_history"] = "other"
    with pytest.raises(ValueError):
        abstain_questions(positive, negative)


def test_express_duplicate_ids_fail():
    row = {"episode_id": 0, "question": "What?", "answer": "blue", "type": "attribute", "scene_id": "scene"}
    with pytest.raises(ValueError, match="duplicate"):
        express_questions([row, row])


def test_upstream_unlabeled_category_preserves_pair_but_is_not_pilot_category():
    rows = [abstain_row()]
    negative = [abstain_row()]
    negative[0]["category"] = float("nan")
    questions = abstain_questions(rows, negative)
    assert len(questions) == 2 and questions[1].category == "unlabeled"
    with pytest.raises(ValueError, match="categorized"):
        development_pilot(questions)


def test_abstention_variants_intentionally_differ_in_question_and_category():
    original, ambiguous = abstain_row(), abstain_row()
    original["question"] = "What is the white object above the TV?"
    original["category"] = "object recognition"
    paired = abstain_questions([original], [ambiguous])
    assert len(development_pilot(paired, 1)) == 2
    assert paired[0].scene is None
    assert paired[0].evidence_id == paired[1].evidence_id == "scene"


def test_mixed_benchmark_pilot_fails_instead_of_guessing_variant_policy():
    express = express_questions(
        [{"episode_id": 0, "question": "What?", "answer": "blue", "type": "attribute", "scene_id": "scene"}]
    )
    abstain = abstain_questions([abstain_row()], [abstain_row()])
    with pytest.raises(ValueError, match="homogeneous"):
        development_pilot(express + abstain, 1)
