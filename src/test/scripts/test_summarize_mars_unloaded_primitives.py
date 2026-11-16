# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Readout evidence detects physical drift and refuses corrupted raw rows."""

import copy
import gzip
import hashlib
import json
import runpy
from pathlib import Path

import pytest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value).encode()
    path.write_bytes(gzip.compress(data) if path.suffix == ".gz" else data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def sample(tmp_path):
    root = Path(__file__).resolve().parents[3]
    module = runpy.run_path(str(root / "scripts/summarize_mars_unloaded_primitives.py"))
    run = tmp_path / "raw"
    baseline = tmp_path / "baseline.json"
    row = dict.fromkeys(module["PHYSICAL_FIELDS"])
    row.update(id="case", code="pregrasp_complete", returns=[1, 2, 3])
    write(baseline, {"teacher": {"tamp": [copy.deepcopy(row)]}})
    manifest_sha = write(run / "sources.json", {"source": "frozen"})
    stages = {}
    for name in ("pregrasp", "grasp_pose"):
        stages[name] = {"rows_file_sha256": write(run / (name + ".json"), [row])}
        write(
            run / name / "episode-0001.json.gz",
            {
                "case": {"id": "case"},
                "ticks": [{"time": 1.0, "base": [0, 0, 0], "policy_inference_s": 0.1}],
                "stop_confirmed": True,
                "commands": [{"phase": "grasp", "sim_time": 1.0}] if name == "grasp_pose" else [],
            },
        )
    write(run / "results.json", {"complete": True, "source_manifest_sha256": manifest_sha, "stages": stages})
    return module["summarize"], run, baseline, tmp_path / "archive"


@pytest.mark.parametrize("physical_drift", [False, True])
def test_prefix_comparison_excludes_wall_time_and_detects_physical_drift(sample, physical_drift):
    summarize, run, baseline, output = sample
    path = run / "grasp_pose/episode-0001.json.gz"
    trace = json.loads(gzip.decompress(path.read_bytes()))
    trace["ticks"][0]["policy_inference_s"] = 9.0
    trace["ticks"][0]["base"][0] = 0.01 if physical_drift else 0
    write(path, trace)
    result = summarize(run, baseline, output)
    assert result["all_paired_prefixes_equal"] is not physical_drift
    assert result["all_pregrasp_physical_fields_unchanged"]
    assert json.loads((output / "results.json").read_text())["paired_prefixes"][0]["prefix_equal"] is not physical_drift


def test_baseline_drift_is_reported_without_relabeling_current_result(sample):
    summarize, run, baseline, output = sample
    original = json.loads(baseline.read_text())
    original["teacher"]["tamp"][0]["returns"][0] = 99
    write(baseline, original)
    result = summarize(run, baseline, output)
    assert not result["all_pregrasp_physical_fields_unchanged"]
    assert result["baseline_physical_parity"][0]["changed_physical_fields"] == ["returns"]


def test_corrupted_rows_cannot_produce_an_archive(sample):
    summarize, run, baseline, output = sample
    (run / "grasp_pose.json").write_text("[]")
    with pytest.raises(ValueError, match="row hash mismatch"):
        summarize(run, baseline, output)
    assert not output.exists()
