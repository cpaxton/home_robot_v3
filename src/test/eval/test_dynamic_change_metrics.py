# Copyright (c) Chris Paxton 2026

import math

from emet.eval.dynamic_change_metrics import (
    change_conditioned_answer_accuracy,
    score_hidden_relocations,
    stale_memory_half_life,
)


def test_hidden_relocation_metrics():
    moves = [
        {
            "old_pos": [0.0, 0.0, 0.5],
            "verified_pos": [2.0, 0.0, 0.5],
            "step": 10,
        }
    ]
    events = [
        {
            "type": "position_contradiction",
            "from_xyz": [0.1, 0.0, 0.5],
            "to_xyz": [2.1, 0.0, 0.5],
            "step": 13,
        },
        {"type": "expected_object_missing", "last_xyz": [8.0, 8.0, 0.0], "step": 12},
    ]
    metrics = score_hidden_relocations(events, moves)
    assert metrics["detection_recall"] == 1.0
    assert metrics["detection_precision"] == 0.5
    assert metrics["false_invalidations"] == 1
    assert metrics["mean_detection_delay_steps"] == 3.0
    assert abs(metrics["mean_relocation_error_m"] - 0.1) < 1e-6


def test_stale_half_life_and_change_answer_accuracy():
    assert stale_memory_half_life([8, 6, 4, 2], steps=[0, 2, 4, 6]) == 4.0
    assert math.isinf(stale_memory_half_life([8, 7, 6]))
    assert (
        change_conditioned_answer_accuracy(
            [
                {"change_expected": True, "answer_correct": True},
                {"change_expected": True, "answer_correct": False},
                {"change_expected": False, "answer_correct": False},
            ]
        )
        == 0.5
    )
