# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from pathlib import Path

import pytest

from emet.habitat.datasets import get_question, load_hmeqa_questions, load_scene_init_poses

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_load_hmeqa_fixture_questions():
    qs = load_hmeqa_questions(FIXTURES / "questions.csv")
    assert len(qs) == 1
    assert qs[0].scene == "00004-VqCaAuuoeWk"
    assert qs[0].answer_letter == "B"
    assert qs[0].answer_index == 1
    assert len(qs[0].choices) == 4


def test_load_scene_init_poses_fixture():
    poses = load_scene_init_poses(FIXTURES / "scene_init_poses.csv")
    key = ("00004-VqCaAuuoeWk", 1)
    assert key in poses
    assert poses[key].x == 0.0


def test_get_question_by_id():
    qs = load_hmeqa_questions(FIXTURES / "questions.csv")
    q = get_question(qs, question_id=0)
    assert "lamp" in q.question.lower()


def test_get_question_missing_file():
    with pytest.raises(FileNotFoundError):
        load_hmeqa_questions(FIXTURES / "missing.csv")


def test_load_scene_init_poses_graph_eqa_format(tmp_path: Path):
    """Explore-EQA CSV uses scene_floor,init_x,init_y,init_z,init_angle."""
    csv_path = tmp_path / "poses.csv"
    csv_path.write_text(
        "scene_floor,init_x,init_y,init_z,init_angle\n"
        "00004-VqCaAuuoeWk_1,1.0,2.0,0.0,0.5\n",
        encoding="utf-8",
    )
    poses = load_scene_init_poses(csv_path)
    key = ("00004-VqCaAuuoeWk", 1)
    assert key in poses
    assert poses[key].x == 1.0
    assert poses[key].heading == 0.5
