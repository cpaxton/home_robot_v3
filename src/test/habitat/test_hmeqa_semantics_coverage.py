# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from pathlib import Path

from emet.habitat.hm3d_semantics import (
    compute_hmeqa_semantics_coverage,
    format_hmeqa_semantics_coverage_report,
    hmeqa_annotated_question_ids,
)


def _write_questions_csv(path: Path) -> None:
    path.write_text(
        "scene,floor,question,choices,question_formatted,answer,label\n"
        "00001-SceneA,1,q1,[],q1,A,\n"
        "00002-SceneB,1,q2,[],q2,B,\n"
        "00003-SceneC,1,q3,[],q3,C,\n",
        encoding="utf-8",
    )


def test_hmeqa_semantics_coverage_counts(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    hm3d_data = tmp_path / "hm3d"
    train = hm3d_data / "scene_datasets" / "hm3d" / "train"
    _write_questions_csv(data_dir / "questions.csv")

    scene_a = train / "00001-SceneA"
    scene_b = train / "00002-SceneB"
    scene_c = train / "00003-SceneC"
    for scene in (scene_a, scene_b, scene_c):
        scene.mkdir(parents=True)
        short = scene.name.split("-", 1)[1]
        (scene / f"{short}.basis.glb").write_bytes(b"x")
    (scene_a / "SceneA.semantic.glb").write_bytes(b"x")

    cov = compute_hmeqa_semantics_coverage(
        hm3d_root=train,
        hm3d_data_root=hm3d_data,
        questions_path=data_dir / "questions.csv",
        paper_question_count=3,
    )
    assert cov.questions_with_semantics == (0,)
    assert cov.questions_without_semantics == (1, 2)
    assert cov.scenes_with_semantics == ("00001-SceneA",)
    assert cov.scenes_without_semantics == ("00002-SceneB", "00003-SceneC")
    assert hmeqa_annotated_question_ids(
        hm3d_root=train,
        questions_path=data_dir / "questions.csv",
        paper_question_count=3,
    ) == [0]

    report = format_hmeqa_semantics_coverage_report(cov)
    assert "with HM3D GT semantics" in report
    assert "fetch-hm3d-semantics" in report
