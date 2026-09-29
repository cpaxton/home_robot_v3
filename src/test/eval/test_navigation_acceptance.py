# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import math

import pytest

from emet.eval.navigation_acceptance import score_arrival_dwell


def samples(poses):
    return [{"time": i * 0.3, "sequence": i, "pose": pose} for i, pose in enumerate(poses)]


def test_fresh_stable_endpoint():
    result = score_arrival_dwell([1, 2, math.pi], samples([[1.001, 2, -math.pi]] * 4), "precision")
    assert result["status"] == "passed_endpoint_dwell"


@pytest.mark.parametrize("pose", [[0.021, 0, 0], [0, 0, 0.031]])
def test_receipt_cannot_hide_endpoint_error(pose):
    assert score_arrival_dwell([0, 0, 0], samples([pose] * 4), "precision")["reason"] == "outside_arrival_tolerance"


def test_returning_oscillation_is_not_settling():
    result = score_arrival_dwell([0, 0, 0], samples([[0, 0, 0], [0.015, 0, 0], [0, 0, 0]]), "precision")
    assert result["reason"] == "post_arrival_motion"


def test_republished_same_observation_is_not_fresh():
    rows = samples([[0, 0, 0]] * 4)
    for row in rows:
        row["sequence"] = 1
    assert score_arrival_dwell([0, 0, 0], rows, "precision")["status"] == "incomplete_telemetry"


@pytest.mark.parametrize("value", [None, [float("nan"), 0, 0], [0, 0]])
def test_invalid_pose_cannot_pass(value):
    rows = samples([[0, 0, 0]] * 4)
    rows[1]["pose"] = value
    assert score_arrival_dwell([0, 0, 0], rows, "precision")["status"] == "incomplete_telemetry"
