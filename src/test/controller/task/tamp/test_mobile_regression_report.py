# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Incomplete cases, negative controls and native crashes keep distinct accounting."""

import json
from pathlib import Path

import pytest


def test_report_preserves_declared_denominator_and_separates_negative_controls(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[5] / "scripts"))
    from report_mobile_map_regression import summarize

    cases = [{"id": "done"}, {"id": "crash"}, {"id": "queued"}, {"id": "fault", "fault": "unknown_after_grasp"}]
    rows = [
        {
            "id": c["id"],
            "case": c,
            "status": "passed" if c["id"] != "crash" else "failed",
            "exit_code": -11 if c["id"] == "crash" else 0,
            "result": {"passed": True},
        }
        for c in cases
        if c["id"] != "queued"
    ]
    (tmp_path / "lane-0.json").write_text(json.dumps({"cases": rows}))
    result = summarize({"scope": "test", "cases": cases}, tmp_path)
    assert (result["task_passed"], result["task_declared"]) == (1, 3)
    assert result["negative_controls_verified"] == result["negative_controls_declared"] == 1
    assert result["unfinished"] == 1
    assert [row["status"] for row in result["cases"]] == ["passed", "native_crash", "pending", "passed"]
    assert result["benchmark_denominator"] is None


def test_duplicate_case_receipts_reject_instead_of_double_count(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[5] / "scripts"))
    from report_mobile_map_regression import summarize

    case = {"id": "one"}
    data = {"cases": [{"id": "one", "case": case, "status": "passed"}]}
    for i in (0, 1):
        (tmp_path / f"lane-{i}.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="inconsistent_regression_manifest"):
        summarize({"scope": "test", "cases": [case]}, tmp_path)
