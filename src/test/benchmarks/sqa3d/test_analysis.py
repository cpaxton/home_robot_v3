# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from pathlib import Path

from emet.benchmarks.sqa3d.analysis import (
    classify_outcome,
    dedupe_episodes_by_question_id,
    generate_sqa3d_figure_bundle,
    summarize_outcomes,
)


def test_classify_outcome():
    assert classify_outcome(em=True, prediction="white", prediction_clean="white") == "tp"
    assert classify_outcome(em=False, prediction="red", prediction_clean="red") == "fp"
    assert classify_outcome(em=False, prediction="unknown", prediction_clean="unknown") == "fn"
    assert (
        classify_outcome(em=False, prediction="CUDA out of memory", prediction_clean="cuda out of memory")
        == "infra"
    )
    assert (
        classify_outcome(em=False, prediction="", prediction_clean="", infra_failure=True)
        == "infra"
    )


def test_dedupe_episodes_last():
    rows = [
        {"question_id": 1, "predicted_answer": "a"},
        {"question_id": 2, "predicted_answer": "b"},
        {"question_id": 1, "predicted_answer": "c"},
    ]
    out = dedupe_episodes_by_question_id(rows, keep="last")
    assert len(out) == 2
    by_id = {r["question_id"]: r["predicted_answer"] for r in out}
    assert by_id[1] == "c"
    assert by_id[2] == "b"


def test_summarize_outcomes_infra_failure_flag():
    episodes = [
        {
            "question_id": 1,
            "scene_id": "scene0000_00",
            "question": "What color is the door?",
            "situation": "s",
            "gold_answers": ["white"],
            "predicted_answer": "",
            "em": False,
            "em_refined": False,
            "infra_failure": True,
            "method": "dynagraph",
            "planning_steps": 0,
        },
    ]
    summary = summarize_outcomes(episodes)
    assert summary["n_infra"] == 1
    assert summary["n_scored"] == 0


def test_summarize_outcomes_fixture():
    episodes = [
        {
            "question_id": 1,
            "scene_id": "scene0000_00",
            "question": "What color is the door?",
            "situation": "s",
            "gold_answers": ["white"],
            "predicted_answer": "white",
            "em": True,
            "em_refined": True,
            "confident": True,
            "method": "dynagraph",
            "planning_steps": 10,
        },
        {
            "question_id": 2,
            "scene_id": "scene0000_00",
            "question": "Is the lamp on?",
            "situation": "s",
            "gold_answers": ["yes"],
            "predicted_answer": "no",
            "em": False,
            "em_refined": False,
            "confident": True,
            "method": "dynagraph",
            "planning_steps": 12,
        },
        {
            "question_id": 3,
            "scene_id": "scene0000_00",
            "question": "How many chairs?",
            "situation": "s",
            "gold_answers": ["two"],
            "predicted_answer": "unknown",
            "em": False,
            "em_refined": False,
            "confident": False,
            "method": "dynagraph",
            "planning_steps": 8,
        },
    ]
    summary = summarize_outcomes(episodes)
    assert summary["tp"] == 1
    assert summary["fp"] == 1
    assert summary["fn"] == 1
    assert summary["em@1"] == 1 / 3


def test_generate_figure_bundle(tmp_path: Path):
    jsonl = tmp_path / "episodes.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                '{"question_id":1,"scene_id":"s","question":"What color?","situation":"x",'
                '"gold_answers":["white"],"predicted_answer":"white","em":true,"em_refined":true,'
                '"confident":true,"method":"dynagraph","planning_steps":5}',
                '{"question_id":2,"scene_id":"s","question":"Is it on?","situation":"x",'
                '"gold_answers":["yes"],"predicted_answer":"no","em":false,"em_refined":false,'
                '"confident":true,"method":"dynagraph","planning_steps":5}',
            ]
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "figs"
    bundle = generate_sqa3d_figure_bundle(jsonl, out_dir, write_plots=True)
    assert (out_dir / "sqa3d_outcomes_summary.json").is_file()
    assert (out_dir / "examples_tp.jsonl").is_file()
    assert (out_dir / "outcomes_bar.png").is_file()
    assert bundle["outcomes"]["tp"] == 1
