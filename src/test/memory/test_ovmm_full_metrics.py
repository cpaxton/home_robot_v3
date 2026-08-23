# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Unit tests for full OVMM pick/place GT scoring (no sim)."""

from __future__ import annotations

from emet.eval.ovmm_full import (
    compute_ovmm_full_metrics,
    score_pick_success,
    score_place_success,
)


def _placements():
    return {
        "table": {"cat": "table", "pos": [0.0, -0.5, 0.5]},
        "cube": {"cat": "blue cube", "pos": [0.5, -0.5, 0.5]},
        "object2": {"cat": "red cylinder", "pos": [0.08, -0.55, 0.6]},
    }


def test_score_pick_success_after_displacement():
    before = _placements()
    after = dict(before)
    after["object2"] = {"cat": "red cylinder", "pos": [0.3, -0.2, 0.8]}
    out = score_pick_success(
        before,
        after,
        object_gt_body="object2",
        start_recep="table",
        radius_m=0.3,
    )
    assert out["pick_success"] is True
    assert out["pick_displacement_m"] > 0.2


def test_score_place_success_near_goal_recep():
    placements = _placements()
    placements["object2"] = {"cat": "red cylinder", "pos": [0.48, -0.52, 0.5]}
    out = score_place_success(
        placements,
        object_gt_body="object2",
        goal_recep="blue cube",
        radius_m=0.3,
    )
    assert out["place_success"] is True
    assert out["place_err_obj_to_recep_m"] < 0.1


def test_score_place_fails_without_improvement():
    placements = _placements()
    out = score_place_success(
        placements,
        object_gt_body="object2",
        goal_recep="blue cube",
        radius_m=0.3,
        placements_before=placements,
    )
    assert out["place_success"] is False


def test_score_place_goal_gt_body_ignores_other_category_matches():
    """Sim teleport targets one receptacle; other ``cab_*`` bodies must not vacuous-fail."""
    before = {
        "obj_main": {"cat": "obj", "pos": [0.0, 0.0, 0.8]},
        "cab_1": {"cat": "cab", "pos": [0.0, 0.0, 1.0]},  # already on a cab in XY
        "cab_main": {"cat": "cab", "pos": [1.0, 0.0, 1.0]},
    }
    after = {
        "obj_main": {"cat": "obj", "pos": [1.0, 0.0, 1.02]},
        "cab_1": {"cat": "cab", "pos": [0.0, 0.0, 1.0]},
        "cab_main": {"cat": "cab", "pos": [1.0, 0.0, 1.0]},
    }
    # Without goal_gt_body: min distance to any cab was already 0 → not improved.
    vacuous = score_place_success(
        after,
        object_gt_body="obj_main",
        goal_recep="cab",
        radius_m=0.5,
        placements_before=before,
    )
    assert vacuous["place_success"] is False
    # With teleport target: improved toward cab_main.
    out = score_place_success(
        after,
        object_gt_body="obj_main",
        goal_recep="cab",
        radius_m=0.5,
        placements_before=before,
        goal_gt_body="cab_main",
    )
    assert out["place_success"] is True
    assert out["gt_recep_bodies"] == ["cab_main"]


def test_score_place_invalid_goal_gt_body_fails_closed():
    placements = _placements()
    placements["object2"] = {"cat": "red cylinder", "pos": [0.48, -0.52, 0.5]}

    out = score_place_success(
        placements,
        object_gt_body="object2",
        goal_recep="blue cube",
        radius_m=0.3,
        goal_gt_body="missing_receptacle",
    )

    assert out["place_success"] is False
    assert out["gt_recep_bodies"] == []


def test_goal_place_xyz_picks_farthest_recep():
    from emet.eval.ovmm_full import _goal_place_xyz

    pl = {
        "obj_main": {"cat": "obj", "pos": [0.0, 0.0, 0.8]},
        "cab_1": {"cat": "cab", "pos": [0.0, 0.0, 1.0]},
        "cab_main": {"cat": "cab", "pos": [2.0, 0.0, 1.0]},
    }
    pos, body = _goal_place_xyz(pl, "cab", object_gt_body="obj_main")
    assert body == "cab_main"
    assert pos is not None
    assert abs(float(pos[0]) - 2.0) < 1e-6


def test_compute_ovmm_full_metrics_all_true():
    out = compute_ovmm_full_metrics(
        find_object_success=True,
        find_recep_success=True,
        pick_success=True,
        place_success=True,
    )
    assert out["ovmm_full_success"] is True
    assert out["ovmm_full_partial"] == 1.0


def test_compute_ovmm_full_metrics_find_only():
    out = compute_ovmm_full_metrics(
        find_object_success=True,
        find_recep_success=False,
        pick_success=None,
        place_success=None,
    )
    assert out["ovmm_full_success"] is None
    assert out["ovmm_full_partial"] == 0.5
