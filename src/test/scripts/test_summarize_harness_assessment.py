# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "configs/eval/harness_assessment_20261119.json"


@pytest.fixture
def report_module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import summarize_harness_assessment

    return summarize_harness_assessment


def test_manifest_has_every_tool_and_preserves_distinct_case_assignments(report_module):
    protocol, cases = report_module.load_manifest(MANIFEST, ROOT)
    assert sum(case["stage"] == "geometry" for case in cases) == 12
    assert sum(case["stage"] == "active" for case in cases) == 12
    assert sum(case["stage"] == "replay" for case in cases) == 12
    assert sum(case["stage"] == "robocasa-preflight" for case in cases) == 12
    assert sum(case["stage"] == "robocasa-model" for case in cases) == 9
    assert any(case["id"] == "hardware-mars" and case["status"] == "blocked" for case in cases)
    assert sum(len(group["tools"]) for group in protocol["groups"]) == 44


def test_manifest_rejects_missing_skill(report_module, tmp_path):
    protocol = json.loads(MANIFEST.read_text())
    protocol["groups"][0]["tools"].remove("describe_scene")
    path = tmp_path / "assessment.json"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="missing=.*describe_scene"):
        report_module.load_manifest(path, ROOT)


def test_transfer_roster_preserves_benchmark_embodiments(report_module):
    roster = ROOT / "configs/eval/observed_transfer_ports.json"
    _, original = report_module.load_manifest(MANIFEST, ROOT)
    _, combined = report_module.load_manifest(MANIFEST, ROOT, transfer_ports=roster)
    new = {(row["stage"], row["id"]): row["adapter"] for row in combined if row not in original}
    assert new["transfer-scripted", "robocasa365-held-out-shaker"] == "pandaomron"
    assert new["transfer-preflight", "emet_tabletop-model-table-transfer"] == "stretch"


def test_results_require_independent_check_and_cannot_overwrite_attempt(report_module, tmp_path):
    artifact = tmp_path / "summary.json"
    artifact.write_text("{}")
    row = {
        "stage": "native",
        "id": "native-cylinder",
        "status": "passed",
        "artifact": artifact.name,
        "source_sha": "source-with-results",
    }
    reports = tmp_path / "results.jsonl"
    reports.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Missing independent check"):
        report_module.export(MANIFEST, [reports], tmp_path / "missing", ROOT)
    reports.write_text(json.dumps({
        **row,
        "independent_check": "supported and settled",
        "wall_s": 0.5,
        "collision_s": 0.1,
        "milestones": {"settled": True, "lifted": None},
    }) + "\n")
    data = report_module.export(MANIFEST, [reports], tmp_path / "figures", ROOT)
    actual = next(item for item in data if item["id"] == "native-cylinder")
    assert actual["status"] == "passed" and actual["artifact_sha256"]
    summary = json.loads((tmp_path / "figures" / "summary.json").read_text())
    assert summary["milestones_by_evidence_tier"][actual["tier"]] == {
        "settled": {"achieved": 1}, "lifted": {"unmeasured": 1},
    }
    assert "collision_s" in (tmp_path / "figures" / "case_outcomes.csv").read_text().splitlines()[0]
    assert next(item for item in data if item["id"] == "stop-only")["status"] == "pending"
    with pytest.raises(ValueError, match="Multiple results"):
        report_module.export(MANIFEST, [reports, reports], tmp_path / "duplicate", ROOT)
