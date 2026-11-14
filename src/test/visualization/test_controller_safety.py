# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import json

import pytest

from emet.visualization.controller_safety import load_report, summarize, wilson, write_report


def test_unknown_support_and_missing_outcomes_do_not_become_safe():
    result = summarize(
        [
            {"is_success": True, "wheel_lift": False},
            {"is_success": False, "wheel_lift": False, "wheel_support_evaluated": False},
        ]
    )
    assert result["wheel_lift"]["count"] == 0
    assert result["unsafe_contact"]["interval"] is None
    assert result["success"]["hits"] == 1
    assert result["success"]["count"] == 2


def test_failed_initialization_is_failure_but_not_safe_execution():
    result = summarize(
        [
            {"initialization_failed": True, "is_success": True, "unsafe_contact": False},
            {"is_success": True, "unsafe_contact": True, "wheel_lift": True, "wheel_support_evaluated": True},
        ]
    )
    assert result["initialization_failures"] == 1
    assert result["success"]["hits"] == 1
    assert result["unsafe_contact"]["hits"] == result["unsafe_contact"]["count"] == 1
    assert result["wheel_lift"]["hits"] == 1


def test_zero_observed_failures_keeps_nonzero_uncertainty():
    assert wilson(0, 0) is None
    assert wilson(0, 20)[1] == pytest.approx(0.161125, abs=1e-6)


def test_reports_preserve_scenarios_scope_and_escape_inline_data(tmp_path):
    source = tmp_path / "report.json"
    source.write_text(
        json.dumps(
            {
                "robot": "stretch",
                "safety_scope": {"wheel_lift_evaluated": True},
                "episodes": [
                    {
                        "seed": 1,
                        "scenario": "carry",
                        "is_success": True,
                        "wheel_lift": False,
                        "time": 1.0,
                        "peak_contact_force": 2.0,
                    },
                    {"seed": 2, "scenario": "reach", "is_success": False},
                ],
            }
        )
    )
    cohorts = load_report("</script><script>alert(1)</script>", source)
    assert len(cohorts) == 2
    assert cohorts[0]["summary"]["wheel_lift"]["count"] == 1
    assert len(cohorts[0]["source"]["sha256"]) == 64
    output = tmp_path / "report.html"
    write_report(cohorts, output)
    document = output.read_text()
    assert "</script><script>alert(1)</script>" not in document
    assert document.count("<svg") == 2
    assert "/*REPORT_DATA*/" not in document
    assert "Unknown (0 evaluated)" in document


def test_summary_report_without_episodes_is_rejected(tmp_path):
    source = tmp_path / "summary.json"
    source.write_text('{"success_rate": 1}')
    with pytest.raises(ValueError, match="episode evaluation"):
        load_report("summary", source)


@pytest.mark.parametrize("metadata", [{"config": None, "safety_scope": None}, {}])
def test_optional_null_metadata_stays_unknown(tmp_path, metadata):
    source = tmp_path / "report.json"
    source.write_text(json.dumps({**metadata, "episodes": [{"is_success": True}]}))
    cohort = load_report("test", source)[0]
    assert cohort["summary"]["wheel_lift"]["count"] == 0
    assert cohort["source"]["force_definition"] == "definition unspecified"


@pytest.mark.parametrize(
    "key,value",
    [
        ("config", []),
        ("safety_scope", "none"),
        ("initialization_failed", 1),
        ("initialization_failed", "true"),
        ("wheel_support_evaluated", 0),
    ],
)
def test_invalid_metadata_and_boolean_types_are_rejected(tmp_path, key, value):
    source = tmp_path / "report.json"
    report = {key: value, "episodes": [{}]} if key in {"config", "safety_scope"} else {"episodes": [{key: value}]}
    source.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=key):
        load_report("invalid", source)


def test_row_support_overrides_report_support_and_preserves_force_definition(tmp_path):
    source = tmp_path / "report.json"
    source.write_text(
        json.dumps(
            {
                "safety_scope": {"wheel_lift_evaluated": True},
                "measurement_definitions": {"peak_contact_force": "L1 linear-force bound"},
                "episodes": [{"wheel_lift": False}, {"wheel_lift": False, "wheel_support_evaluated": False}],
            }
        )
    )
    cohort = load_report("test", source)[0]
    assert cohort["summary"]["wheel_lift"]["count"] == 1
    assert cohort["summary"]["wheel_lift"]["unknown"] == 1
    assert cohort["source"]["measurement_definitions"]["peak_contact_force"] == "L1 linear-force bound"


def test_repeated_trials_are_flagged_without_inflating_independent_evidence():
    result = summarize([{"seed": 1, "is_success": True}, {"seed": 1, "is_success": True}, {}])
    assert result["attempts"] == 3
    assert result["repeated_identities"] == 1
    assert result["missing_identities"] == 1
    assert result["success"]["unknown"] == 1


def test_unique_episode_ids_do_not_hide_identical_case_seed_replays():
    rows = [{"episode_id": f"record-{i}", "case_id": "fixed-case", "seed": 90, "is_success": True} for i in range(100)]
    result = summarize(rows)
    assert result["attempts"] == result["success"]["count"] == 100
    assert result["unique_trial_keys"] == 1
    assert result["repeated_identities"] == 99
    assert result["repeated_record_ids"] == result["missing_identities"] == 0
    assert result["success"]["interval"] is None


def test_duplicate_record_ids_are_separate_from_trial_keys():
    result = summarize(
        [
            {"episode_id": "record-1", "case_id": "case-A", "seed": 1, "is_success": True},
            {"episode_id": "record-1", "case_id": "case-A", "seed": 2, "is_success": True},
        ]
    )
    assert result["unique_trial_keys"] == 2
    assert result["repeated_identities"] == 0
    assert result["repeated_record_ids"] == 1
    assert result["success"]["interval"] is None


def test_distinct_case_seed_pairs_retain_conditional_intervals():
    result = summarize(
        [
            {"episode_id": "record-1", "case_id": "case-A", "seed": 1, "is_success": True},
            {"episode_id": "record-2", "case_id": "case-B", "seed": 1, "is_success": True},
            {"episode_id": "record-3", "case_id": "case-A", "seed": 2, "is_success": True},
        ]
    )
    assert result["unique_trial_keys"] == 3
    assert result["repeated_identities"] == result["repeated_record_ids"] == result["missing_identities"] == 0
    assert result["success"]["interval"] == wilson(3, 3)


def test_episode_id_without_seed_is_not_independent_trial_identity():
    result = summarize([{"episode_id": "record-1", "case_id": "case-A", "is_success": True}])
    assert result["unique_trial_keys"] == 0
    assert result["missing_identities"] == 1
    assert result["success"]["interval"] is None


def test_replayed_trials_show_warning_without_confident_binomial_bound(tmp_path):
    source = tmp_path / "report.json"
    source.write_text(
        json.dumps(
            {
                "episodes": [
                    {"episode_id": f"record-{i}", "case_id": "fixed-case", "seed": 90, "is_success": True}
                    for i in range(100)
                ]
            }
        )
    )
    cohorts = load_report("replays", source)
    output = tmp_path / "viewer.html"
    write_report(cohorts, output)
    document = output.read_text()
    assert "99 repeated trials" in document
    assert "100/100 (CI unavailable: trial independence unverified)" in document
    assert "96%–100% CI" not in document


def test_template_sentinel_in_label_and_script_data_is_preserved(tmp_path):
    source = tmp_path / "report.json"
    source.write_text('{"episodes": [{"is_success": true}]}')
    cohorts = load_report("/*REPORT_DATA*/ null", source)
    output = tmp_path / "viewer.html"
    write_report(cohorts, output)
    document = output.read_text()
    data = document.split("const cohorts = ", 1)[1].split(";\nconst cohortSelect", 1)[0]
    assert json.loads(data)[0]["name"] == cohorts[0]["name"]
    assert "<td>/*REPORT_DATA*/ null / unspecified / unspecified</td>" in document


@pytest.mark.parametrize("episodes", ["[null]", '[{"time": NaN}]'])
def test_invalid_episode_or_nonfinite_json_is_rejected(tmp_path, episodes):
    source = tmp_path / "report.json"
    source.write_text('{"episodes": ' + episodes + "}")
    with pytest.raises(ValueError):
        load_report("invalid", source)
