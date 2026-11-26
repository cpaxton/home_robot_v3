# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Coverage registry, split, and promotion-gate checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emet.rl.manipulation.wbc_coverage import (
    CoverageCase,
    gate_report,
    load_coverage,
)


def _case(case_id, family, split="test", status="active", occupancy="none"):
    return {
        "id": case_id,
        "family": family,
        "robot": "pandaomron",
        "split": split,
        "status": status,
        "source": "test",
        "occupancy": occupancy,
        "gate": {},
    }


def _write(tmp_path, cases):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"schema_version": 1, "cases": cases}), encoding="utf-8")
    return path


def test_registry_requires_a_test_case_for_active_families(tmp_path):
    path = _write(tmp_path, [_case("train", "free_space_reach", split="train")])
    with pytest.raises(ValueError, match="no test case"):
        load_coverage(path)


def test_registry_allows_planned_families_without_tests(tmp_path):
    path = _write(
        tmp_path,
        [
            _case("arm", "free_space_reach"),
            _case("shelf", "shelf_entry", status="planned", occupancy="required"),
        ],
    )
    cases = load_coverage(path)
    assert [case.id for case in cases] == ["arm", "shelf"]
    assert cases[1].occupancy == "required"


def test_duplicate_case_ids_are_rejected(tmp_path):
    path = _write(tmp_path, [_case("arm", "free_space_reach"), _case("arm", "free_space_reach")])
    with pytest.raises(ValueError, match="duplicate"):
        load_coverage(path)


def test_gate_passes_all_active_test_cases():
    case = CoverageCase(
        id="arm",
        family="free_space_reach",
        robot="pandaomron",
        split="test",
        status="active",
        source="test",
        gate={"min_success": 0.9, "max_collision_rate": 0.0, "max_tracking_error_m": 0.03},
    )
    report = gate_report(
        [case],
        {"arm": {"episodes": 10, "success_rate": 0.95, "collisions": 0, "tracking_error_m": 0.02}},
    )
    assert report["passed"] is True
    assert report["families"]["free_space_reach"]["passed"] is True


def test_gate_rejects_missing_results_and_family_regressions():
    case = CoverageCase(
        id="arm",
        family="free_space_reach",
        robot="pandaomron",
        split="test",
        status="active",
        source="test",
    )
    report = gate_report([case], {})
    assert report["passed"] is False
    assert report["missing_results"] == ["arm"]

    report = gate_report(
        [case],
        {"arm": {"episodes": 10, "success_rate": 0.8, "collisions": 0, "tracking_error_m": 0.02}},
    )
    assert report["passed"] is False
    assert any("success" in failure for failure in report["failures"])


def test_panda_hard_cases_are_split_without_leakage():
    train = {case.name for case in load_hard_case_file("configs/eval/wbc_pandaomron_hard_cases_train.json")}
    test = {case.name for case in load_hard_case_file("configs/eval/wbc_pandaomron_hard_cases_test.json")}
    assert len(train) == 5
    assert len(test) == 1
    assert not train & test


def load_hard_case_file(path):
    from emet.rl.manipulation.wbc_hard_cases import load_hard_cases

    return load_hard_cases(path)
