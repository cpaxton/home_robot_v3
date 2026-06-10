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

"""Unit tests for Habitat OVMM find-phase episode loader (no sim)."""

from __future__ import annotations

from emet.eval.habitat_ovmm_find import load_habitat_ovmm_episodes, score_habitat_find_phase


def test_load_habitat_ovmm_episodes():
    rows = load_habitat_ovmm_episodes()
    assert len(rows) >= 1
    assert rows[0]["scene"]


def test_score_habitat_find_phase_frame():
    placements = {
        "hm3d_lamp_1": {
            "cat": "lamp",
            "pos": [1.0, 1.2, 2.0],
            "bounds": [[0.8, 1.0, 1.8], [1.2, 1.4, 2.2]],
            "frame": "habitat_yup",
        },
        "hm3d_bed_2": {
            "cat": "bed",
            "pos": [0.5, 0.0, 1.0],
            "bounds": [[0.0, 0.0, 0.5], [1.0, 0.5, 1.5]],
            "frame": "habitat_yup",
        },
    }
    metrics = score_habitat_find_phase(
        obj_pred_xyz=[1.0, 1.2, 2.0],
        recep_pred_xyz=[0.5, 0.0, 1.0],
        placements=placements,
        object_query="lamp",
        start_recep="bed",
        goal_recep="bed",
        radius_m=0.75,
    )
    assert metrics["find_object_success"] is True
    assert metrics["find_recep_success"] is True
