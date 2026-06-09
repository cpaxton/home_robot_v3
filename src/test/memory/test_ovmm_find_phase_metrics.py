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

"""Unit tests for OVMM find-phase GT oracles (no sim required)."""

from __future__ import annotations

import numpy as np

from emet.eval.ovmm_find_phase import (
    _query_variants,
    category_matches,
    compute_find_phase_metrics,
    distance_to_placement_xy,
    horizontal_coords,
    localization_pred_fields,
    pick_find_object_gt_body,
    pred_xyz_to_json_list,
    score_find_object,
    score_find_recep,
)


def _default_table_placements():
    return {
        "table": {"cat": "table", "pos": [0.0, -0.5, 0.5]},
        "object1": {"cat": "blue cube", "pos": [-0.1, -0.55, 0.6]},
        "object2": {"cat": "red cylinder", "pos": [0.08, -0.55, 0.6]},
    }


def test_category_matches_substring():
    assert category_matches("red cylinder", "red cylinder")
    assert category_matches("red", "red cylinder")
    assert category_matches("cabinet", "kitchen cabinet door")
    assert not category_matches("sofa", "table")


def test_pick_find_object_prefers_start_recep():
    placements = {
        "apple_a": {"cat": "apple", "pos": [1.0, 0.0, 0.5]},
        "apple_b": {"cat": "apple", "pos": [0.05, -0.5, 0.6]},
        "counter": {"cat": "counter", "pos": [0.0, -0.5, 0.9]},
    }
    body = pick_find_object_gt_body(placements, "apple", "counter")
    assert body == "apple_b"


def test_score_find_object_success_within_radius():
    placements = _default_table_placements()
    pred = np.array([0.08, -0.55, 0.6])
    out = score_find_object(
        pred,
        placements,
        "red cylinder",
        "table",
        radius_m=0.75,
    )
    assert out["find_object_success"] is True
    assert out["localization_err_obj_m"] == 0.0
    assert out["gt_object_body"] == "object2"


def test_score_find_object_failure_far():
    placements = _default_table_placements()
    pred = np.array([2.0, 2.0, 0.0])
    out = score_find_object(
        pred,
        placements,
        "red cylinder",
        "table",
        radius_m=0.75,
    )
    assert out["find_object_success"] is False
    assert out["localization_err_obj_m"] > 0.75


def test_score_find_recep_any_matching_body():
    placements = {
        "cab_a": {"cat": "cabinet", "pos": [0.0, 1.0, 0.5]},
        "cab_b": {"cat": "upper cabinet", "pos": [0.5, 1.0, 1.2]},
    }
    pred = np.array([0.1, 1.05, 0.0])
    out = score_find_recep(pred, placements, "cabinet", radius_m=0.75)
    assert out["find_recep_success"] is True
    assert out["localization_err_recep_m"] < 0.2


def test_pick_find_object_respects_gt_body():
    placements = _default_table_placements()
    body = pick_find_object_gt_body(
        placements,
        "wrong query",
        "table",
        object_gt_body="object2",
    )
    assert body == "object2"


def test_query_variants_includes_gt_cat():
    variants = _query_variants("cab", {"cab_main": {"cat": "cab"}})
    assert "cab" in variants


def test_distance_to_bounds_habitat_xz():
    placement = {
        "cat": "table",
        "pos": [1.0, 0.5, 2.0],
        "bounds": [[0.5, 0.0, 1.5], [1.5, 1.0, 2.5]],
        "frame": "habitat_yup",
    }
    err = distance_to_placement_xy([1.0, 0.5, 2.0], placement, frame="habitat_xz")
    assert err == 0.0
    err2 = distance_to_placement_xy([2.0, 0.5, 2.0], placement, frame="habitat_xz")
    assert abs(err2 - 0.5) < 1e-6
    assert horizontal_coords([1.0, 0.5, 2.0], frame="habitat_xz").tolist() == [1.0, 2.0]


def test_pred_xyz_to_json_list():
    assert pred_xyz_to_json_list(None) is None
    assert pred_xyz_to_json_list([0.1, -0.5]) == [0.1, -0.5, 0.0]
    assert pred_xyz_to_json_list(np.array([0.08, -0.55, 0.6])) == [0.08, -0.55, 0.6]
    fields = localization_pred_fields([0.08, -0.55, 0.6], None)
    assert fields["pred_obj_xyz"] == [0.08, -0.55, 0.6]
    assert fields["pred_recep_xyz"] is None


def test_compute_find_phase_partial_success():
    placements = _default_table_placements()
    metrics = compute_find_phase_metrics(
        obj_pred_xyz=[0.08, -0.55, 0.6],
        recep_pred_xyz=[0.0, -0.5, 0.5],
        placements=placements,
        object_query="red cylinder",
        start_recep="table",
        goal_recep="table",
        radius_m=0.75,
    )
    assert metrics["find_object_success"] is True
    assert metrics["find_recep_success"] is True
    assert metrics["find_partial_success"] == 1.0
