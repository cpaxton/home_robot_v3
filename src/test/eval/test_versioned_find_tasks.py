# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from pathlib import Path

import pytest

from emet.eval.ovmm_find_phase import load_find_phase_episodes, resolve_object_query, score_find_recep


def test_versioned_robocasa_task_keeps_public_language_and_scoring_annotations_separate():
    root = Path(__file__).resolve().parents[3]
    episodes = {e.id: e for e in load_find_phase_episodes(root / "configs/ovmm/find_phase_episodes.yaml")}
    assert episodes["robocasa_pp_s1"].object == "jar"  # Retain the invalid historical protocol.
    fixed = episodes["robocasa_pp_s1_v2"]
    assert resolve_object_query(fixed, {"obj_main": {"cat": "sugar_cube"}}) == "sugar cube"
    assert fixed.goal_recep == "cabinet"
    assert len(fixed.goal_recep_gt_bodies) == 6


def test_declared_target_set_accepts_each_valid_instance_and_excludes_decoys():
    placements = {
        name: {"cat": "cab", "pos": xyz}
        for name, xyz in (("one", [0.0, 0.0, 0.0]), ("two", [3.0, 0.0, 0.0]), ("unrelated", [6.0, 0.0, 0.0]))
    }
    for body, x in (("one", 0.0), ("two", 3.0)):
        result = score_find_recep(
            [x, 0.0, 0.0], placements, "cabinet", radius_m=0.5, goal_recep_gt_bodies=("one", "two")
        )
        assert result["find_recep_success"]
        assert result["gt_recep_body"] == body
    assert not score_find_recep(
        [6.0, 0.0, 0.0], placements, "cabinet", radius_m=0.5, goal_recep_gt_bodies=("one", "two")
    )["find_recep_success"]


def test_missing_annotation_never_falls_back_to_an_easier_existing_target():
    result = score_find_recep(
        [0.0, 0.0, 0.0],
        {"one": {"cat": "cab", "pos": [0.0, 0.0, 0.0]}},
        "cabinet",
        radius_m=0.5,
        goal_recep_gt_bodies=("one", "missing"),
    )
    assert not result["find_recep_scored"] and not result["find_recep_success"]
    assert result["find_recep_unscored_reason"] == "missing_annotated_target"


def test_invalid_annotation_list_rejected_before_running_scene(tmp_path):
    path = tmp_path / "tasks.yaml"
    path.write_text(
        "episodes:\n- id: bad\n  sim: scene\n  object: cup\n  start_recep: table\n  goal_recep: shelf\n  goal_recep_gt_bodies: [one, one]\n"
    )
    with pytest.raises(ValueError, match="unique"):
        load_find_phase_episodes(path)
