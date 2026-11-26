# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Coverage registry, split, and promotion-gate checks."""

from __future__ import annotations

import json

import pytest

from emet.rl.manipulation.wbc_coverage import (
    CoverageCase,
    gate_report,
    hard_case_metrics,
    load_coverage,
    task_result_metrics,
)


def _case(case_id, family, split="test", status="active", occupancy="none", contact="none"):
    return {
        "id": case_id,
        "family": family,
        "robot": "pandaomron",
        "split": split,
        "status": status,
        "source": "test",
        "occupancy": occupancy,
        "contact": contact,
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
        {"arm": {"episodes": 10, "success_rate": 0.95, "collisions": 0, "tracking_error_m": 0.02, "duration_s": 5.0}},
    )
    assert report["passed"] is True
    assert report["families"]["free_space_reach"]["passed"] is True


def test_gate_requires_measured_contact_state_for_contact_families():
    case = CoverageCase(
        id="contact",
        family="contact_recovery",
        robot="pandaomron",
        split="test",
        status="active",
        source="test",
        contact="required",
    )
    missing = gate_report(
        [case],
        {
            "contact": {
                "episodes": 10,
                "success_rate": 0.9,
                "collisions": 0,
                "tracking_error_m": 0.02,
                "duration_s": 5.0,
            }
        },
    )
    assert missing["passed"] is False
    assert any("contact state missing" in failure for failure in missing["failures"])

    passed = gate_report(
        [case],
        {
            "contact": {
                "episodes": 10,
                "success_rate": 0.9,
                "collisions": 0,
                "tracking_error_m": 0.02,
                "duration_s": 5.0,
                "contact_state_rate": 1.0,
            }
        },
    )
    assert passed["passed"] is True

    absent = gate_report(
        [case],
        {
            "contact": {
                "episodes": 10,
                "success_rate": 0.9,
                "collisions": 0,
                "tracking_error_m": 0.02,
                "duration_s": 5.0,
                "contact_state_rate": 0.0,
            }
        },
    )
    assert absent["passed"] is False
    assert any("contact state rate" in failure for failure in absent["failures"])


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
        {"arm": {"episodes": 10, "success_rate": 0.8, "collisions": 0, "tracking_error_m": 0.02, "duration_s": 5.0}},
    )
    assert report["passed"] is False
    assert any("success" in failure for failure in report["failures"])


def _gate_case():
    return CoverageCase(
        id="arm",
        family="free_space_reach",
        robot="pandaomron",
        split="test",
        status="active",
        source="test",
    )


@pytest.mark.parametrize("field", ["success_rate", "tracking_error_m", "duration_s"])
def test_gate_rejects_nonfinite_metrics(field):
    result = {"episodes": 10, "success_rate": 0.95, "collisions": 0, "tracking_error_m": 0.02, "duration_s": 5.0}
    result[field] = float("nan")
    report = gate_report([_gate_case()], {"arm": result})
    assert report["passed"] is False
    assert any("finite" in failure or "must be" in failure for failure in report["failures"])


def test_gate_rejects_non_scalar_episodes_and_missing_tracking():
    report = gate_report(
        [_gate_case()],
        {"arm": {"episodes": [10], "success_rate": 0.95, "collisions": 0, "tracking_error_m": 0.02, "duration_s": 5.0}},
    )
    assert report["passed"] is False
    assert any("episodes must be a positive integer" in failure for failure in report["failures"])
    report = gate_report(
        [_gate_case()],
        {"arm": {"episodes": 10, "success_rate": 0.95, "collisions": 0, "duration_s": 5.0}},
    )
    assert report["passed"] is False
    assert any("tracking_error_m" in failure for failure in report["failures"])


def test_coverage_result_rejects_bad_values():
    from emet.rl.manipulation.wbc_coverage import coverage_result

    result = coverage_result(
        episodes=10,
        success_rate=0.9,
        collisions=0,
        tracking_error_m=0.02,
        duration_s=5.0,
        contact_state_rate=1.0,
    )
    assert result["episodes"] == 10
    with pytest.raises(ValueError, match="finite"):
        coverage_result(episodes=10, success_rate=float("nan"), tracking_error_m=0.02, duration_s=5.0)
    with pytest.raises(ValueError, match="positive integer"):
        coverage_result(episodes=0, success_rate=0.9, tracking_error_m=0.02, duration_s=5.0)


def test_task_result_metrics_aggregate_seeds():
    seed_results = {
        "1": {
            "arm": {
                "episodes": 10.0,
                "success_rate": 0.9,
                "collisions": 1,
                "tracking_error_m": 0.02,
                "duration_s": 5.0,
                "contact_state_rate": 0.5,
            }
        },
        "2": {
            "arm": {
                "episodes": 10.0,
                "success_rate": 1.0,
                "collisions": 0,
                "tracking_error_m": 0.04,
                "duration_s": 7.0,
                "contact_state_rate": 1.0,
            }
        },
    }
    row = task_result_metrics(seed_results, "arm")
    assert row["episodes"] == 20
    assert row["success_rate"] == pytest.approx(0.95)
    assert row["collisions"] == 1
    assert row["tracking_error_m"] == pytest.approx(0.03)
    assert row["duration_s"] == pytest.approx(6.0)
    assert row["contact_state_rate"] == pytest.approx(0.75)


def test_hard_case_metrics_accepts_episode_list_and_fails_missing():
    row = hard_case_metrics(
        {
            "episodes": [{"success": True}, {"success": False}],
            "success_rate": 0.5,
            "collisions": 1,
            "tracking_error_m": 0.04,
            "duration_s": 6.0,
        }
    )
    assert row["episodes"] == 2
    with pytest.raises(KeyError):
        task_result_metrics({"1": {"arm": {"episodes": 10}}}, "arm")


def test_panda_hard_cases_are_split_without_leakage():
    train = {case.name for case in load_hard_case_file("configs/eval/wbc_pandaomron_hard_cases_train.json")}
    test = {case.name for case in load_hard_case_file("configs/eval/wbc_pandaomron_hard_cases_test.json")}
    # The 2026-11-25 sink only contains one genuine reached=False failure.
    # Stale place failures (reached=True) are rejected by the exporter.
    assert train == set()
    assert len(test) == 1
    assert not train & test


def load_hard_case_file(path):
    from emet.rl.manipulation.wbc_hard_cases import load_hard_cases

    return load_hard_cases(path)
