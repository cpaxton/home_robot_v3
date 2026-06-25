# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from pathlib import Path

from click.testing import CliRunner

from emet.app.eval_sqa3d import eval_sqa3d_main, sqa3d_group
from emet.benchmarks.sqa3d.scannet.config import filter_questions_with_scannet
from emet.llms.graph_eqa_vlm import _eqa_system_prompt

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_sqa3d_prompt_variant():
    prompt = _eqa_system_prompt({"eqa": {"prompt_variant": "sqa3d"}})
    assert "Situation" in prompt or "situated" in prompt.lower()


def test_eval_sqa3d_fixture_split(tmp_path: Path):
    preds = tmp_path / "preds.jsonl"
    preds.write_text(
        '{"question_id": 220602000000, "answer": "brown"}\n'
        '{"question_id": 220602000001, "answer": "yes"}\n'
        '{"question_id": 220602000002, "answer": "two"}\n',
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        eval_sqa3d_main,
        [
            "-p",
            str(preds),
            "--split",
            "val",
            "--questions-path",
            str(FIXTURES / "v1_balanced_questions_val_scannetv2.json"),
            "--annotations-path",
            str(FIXTURES / "v1_balanced_sqa_annotations_val_scannetv2.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "em@1" in result.output
    assert "1.0" in result.output


def test_sqa3d_info_cli():
    runner = CliRunner()
    result = runner.invoke(sqa3d_group, ["info"])
    assert result.exit_code == 0
    assert "SQA3D_DATA_DIR" in result.output
    assert "SCANNET_ROOT" in result.output


def test_sqa3d_verify_cli():
    runner = CliRunner()
    result = runner.invoke(sqa3d_group, ["verify", "--split", "val"])
    # Passes when default cache has SQA3D + at least smoke scenes; skip in minimal env.
    if result.exit_code != 0:
        import pytest

        pytest.skip(f"sqa3d verify needs cached data: {result.output[:200]}")
    assert "VERIFY OK" in result.output


def test_filter_questions_with_scannet(tmp_path: Path):
    from emet.benchmarks.sqa3d.datasets import load_sqa3d_questions

    qs = load_sqa3d_questions(
        "val",
        questions_path=FIXTURES / "v1_balanced_questions_val_scannetv2.json",
        annotations_path=FIXTURES / "v1_balanced_sqa_annotations_val_scannetv2.json",
    )
    # No meshes under empty tmp_path -> filter removes all
    assert filter_questions_with_scannet(qs, tmp_path) == []
