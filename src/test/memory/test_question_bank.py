# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from __future__ import annotations

from pathlib import Path

from emet.memory.graph_eqa.question_bank import load_question_bank, score_eqa_results


def test_load_all_environments():
    path = Path(__file__).resolve().parents[2] / "emet/config/benchmarks/dynagraph_questions.yaml"
    bank = load_question_bank(path)
    envs = {q["env"] for q in bank}
    assert "default_table" in envs
    assert "robocasa_seed0" in envs


def test_score_partial_tokens_fails():
    scored = score_eqa_results(
        [{"question": "colors?", "answer": "red only", "expected_tokens": ["red", "blue"]}]
    )
    assert scored["accuracy"] == 0.0
