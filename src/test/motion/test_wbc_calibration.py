# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Failure-record calibration and feedforward integration tests."""

from __future__ import annotations

import json

import pytest

from emet.controller.manipulation.wbc_feedforward import FeedforwardModel
from emet.rl.manipulation.wbc_calibration import fit_records, main


def _attempt(bias: float) -> dict:
    target = 0.5
    observed = target - bias
    return {
        "context": {"robot_id": "innate_mars"},
        "attempt": {
            "failure_mode": "tracking_failed",
            "joint_names": ["joint3"],
            "planned_waypoints": [[0.0], [target]],
            "trace": [
                {"qpos": [observed - 0.01]},
                {"qpos": [observed]},
                {"qpos": [observed]},
                {"qpos": [observed]},
            ],
        },
    }


def _write_records(path, biases):
    path.write_text("".join(json.dumps(_attempt(bias)) + "\n" for bias in biases), encoding="utf-8")


def test_fit_records_uses_robust_median_bias(tmp_path):
    records = tmp_path / "records.jsonl"
    _write_records(records, (0.10, 0.20, 0.30))
    model = fit_records(records, embodiment="innate_mars", joint_names=("joint3",), min_samples=2)
    assert model.joint_bias["joint3"] == pytest.approx(0.20)
    assert model.samples == 3


def test_calibration_cli_round_trip(tmp_path):
    records = tmp_path / "records.jsonl"
    _write_records(records, (0.10, 0.20))
    output = tmp_path / "feedforward.json"
    assert (
        main(
            [
                "--records",
                str(records),
                "--output",
                str(output),
                "--embodiment",
                "innate_mars",
                "--min-samples",
                "2",
            ]
        )
        == 0
    )
    loaded = FeedforwardModel.load(output)
    assert loaded.joint_bias["joint3"] == pytest.approx(0.15)
    assert loaded.source == "manip_failures"


def test_fit_records_fails_closed_without_usable_samples(tmp_path):
    records = tmp_path / "records.jsonl"
    records.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="enough valid"):
        fit_records(records, embodiment="innate_mars", min_samples=2)
