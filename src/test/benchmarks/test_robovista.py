# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

from pathlib import Path

from PIL import Image

from emet.benchmarks.robovista.datasets import RoboVistaQuestion, load_robovista
from emet.benchmarks.robovista.metrics import summarize_robovista_rows
from emet.benchmarks.robovista.prompts import build_robovista_prompt, format_choices_block
from emet.benchmarks.robovista.runner import run_robovista_batch


def _fake_question(
    *,
    qid: str = "rv_001",
    domain: str = "domestic",
    gold: str = "B",
    ability: str = "perception",
) -> RoboVistaQuestion:
    img = Image.new("RGB", (16, 16), color=(40, 80, 120))
    return RoboVistaQuestion(
        id=qid,
        question="Which object is closest to the robot?",
        choices=["mug", "bowl", "chair", "lamp", "none"],
        gold_letter=gold,
        domain=domain,
        task="home tidying",
        ability_type=ability,
        images=[img],
    )


def test_format_choices_block_includes_e():
    block = format_choices_block(["a", "b", "c", "d", "e"])
    assert "A. a" in block
    assert "E. e" in block


def test_build_robovista_prompt_letter_only():
    q = _fake_question()
    prompt = build_robovista_prompt(q)
    assert "Question:" in prompt
    assert "Choices:" in prompt
    assert "E. none" in prompt
    assert "Answer:" in prompt
    assert "letter" in prompt.lower()


def test_load_robovista_from_rows_filters_domain():
    rows = [
        {
            "id": "1",
            "question": "Q1?",
            "choices": ["A. x", "B. y", "C. z", "D. w", "E. v"],
            "correct_answer": "C",
            "domain": "domestic",
            "task": "tidy",
            "ability_type": "perception",
            "images": [Image.new("RGB", (8, 8), color=0)],
        },
        {
            "id": "2",
            "question": "Q2?",
            "choices": ["x", "y", "z", "w", "v"],
            "correct_answer": "A",
            "domain": "surgical",
            "task": "knot",
            "ability_type": "planning",
            "images": [Image.new("RGB", (8, 8), color=1)],
        },
    ]
    qs = load_robovista(rows=rows, domains=["domestic"])
    assert len(qs) == 1
    assert qs[0].id == "1"
    assert qs[0].choices[0] == "x"
    assert qs[0].gold_letter == "C"


def test_run_robovista_batch_mock_llm(tmp_path: Path):
    questions = [
        _fake_question(qid="a", gold="C", domain="domestic"),
        _fake_question(qid="b", gold="E", domain="industrial", ability="planning"),
    ]
    summary = run_robovista_batch(
        questions=questions,
        mock_llm=True,
        output_dir=tmp_path / "out",
    )
    assert summary["n"] == 2
    assert summary["correct"] == 2
    assert summary["accuracy"] == 1.0
    assert summary["by_domain"]["domestic"]["n"] == 1
    assert summary["by_domain"]["industrial"]["correct"] == 1
    assert (tmp_path / "out" / "predictions.jsonl").is_file()
    assert (tmp_path / "out" / "summary.json").is_file()


def test_run_robovista_batch_resume(tmp_path: Path):
    out = tmp_path / "resume"
    q1 = _fake_question(qid="keep", gold="A")
    q2 = _fake_question(qid="new", gold="B")
    run_robovista_batch(questions=[q1], mock_llm=True, output_dir=out)
    summary = run_robovista_batch(
        questions=[q1, q2],
        mock_llm=True,
        output_dir=out,
        resume=True,
    )
    assert summary["n"] == 2
    ids = [line for line in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(ids) == 2


def test_summarize_robovista_rows():
    rows = [
        {"domain": "domestic", "ability_type": "perception", "correct": True},
        {"domain": "domestic", "ability_type": "planning", "correct": False},
        {"domain": "industrial", "ability_type": "perception", "correct": True},
    ]
    summary = summarize_robovista_rows(rows)
    assert summary["n"] == 3
    assert summary["correct"] == 2
    assert summary["by_domain"]["domestic"]["accuracy"] == 0.5
