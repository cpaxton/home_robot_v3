# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Prevent population/scoring shortcuts in the serial benchmark refresh."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("benchmark_refresh", SCRIPTS / "run_existing_benchmark_sweep.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_matrix_preserves_gate_population_and_learned_rooms(tmp_path):
    root = SCRIPTS.parent
    matrix = driver.build_matrix(root, tmp_path, "python", "habitat")
    assert len({t["id"] for t in matrix}) == len(matrix)
    eqa = [t for t in matrix if t["family"] == "eqa"]
    assert len(eqa) == 39 * 4
    populations = [{t["command"][3] for t in eqa if f"eqa-{m}-" in t["id"]} for m in driver.METHODS]
    assert all(p == populations[0] for p in populations)
    assert "104" in populations[0] and "105" in populations[0]
    for t in matrix:
        if t["family"] == "room_find":
            assert "--sensor-perception" in t["command"]
            assert "--no-scene-cache" in t["command"]
        if t["family"] == "sqa3d_recorded":
            assert "--no-download" in t["command"] and "sens" in t["command"]


@pytest.mark.parametrize("payload", [None, {}, {"correct": "true"}, {"correct": True, "error": "crash"}])
def test_infrastructure_is_not_scored_as_task_success(tmp_path, payload):
    metric = tmp_path / "metrics.jsonl"
    if payload is not None:
        metric.write_text(json.dumps(payload) + "\n")
    out = driver.read_outcome({"metric": str(metric), "family": "eqa"}, 0)
    assert out["complete"] is False and out["success"] is None


def test_wrong_answer_is_completed_task_not_infrastructure_failure(tmp_path):
    metric = tmp_path / "metrics.jsonl"
    metric.write_text(json.dumps({"correct": False}) + "\n")
    out = driver.read_outcome({"metric": str(metric), "family": "eqa"}, 0)
    assert out["complete"] is True and out["success"] is False and out["error"] is None
