"""Structural evidence-gate tests; synthetic reports are not physics results."""

import copy
import hashlib
import json

import numpy as np
import pytest

from emet.eval import mars_tamp_control as benchmark


@pytest.fixture
def report(monkeypatch):
    monkeypatch.setattr(benchmark, "sources", lambda: {"test_manifest": "current"})
    catalog = json.loads(benchmark.CANONICAL_SUITE.read_text())
    return {
        "suite": catalog,
        "suite_sha256": hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest(),
        "source_hashes": benchmark.sources(),
        "complete": True,
        "cases": [
            {
                "case": case,
                "success": True,
                "selector": "oracle",
                "input_mode": "privileged",
                "fixture_admission": {"accepted": True},
                "frozen_fixture_verified": True,
                "model_signature": case["model_signature"],
                "post_execution_stable_s": 2.0,
                "final_predicates": dict.fromkeys(
                    [
                        "inside_goal_region",
                        "destination_support",
                        "gripper_open",
                        "fingers_clear",
                        "object_still",
                        "safe",
                        "robot_still",
                        "base_goal_within_tolerance",
                    ],
                    True,
                ),
                "peak_base_displacement_m": 0.1,
                "max_payload_rotation_rad": 0.02,
                "max_payload_translation_m": 0.001,
                "robot_failure": None,
                "contact_metrics": {"wheel_lift": False, "wheel_support_evaluated": True, "peak_contact_force": 3.0},
                "calls": [
                    {
                        "tool": "execute_pick_place_plan",
                        "result": {"status": "ok", "data": {"completed_ops": ["grasp", "place"]}},
                    }
                ],
            }
            for case in catalog["cases"]
        ],
    }


def test_catalog_has_all_mass_layout_seed_identities_and_accepted_witnesses():
    catalog = json.loads(benchmark.CANONICAL_SUITE.read_text())
    cases = catalog["cases"]
    assert len(cases) == len({c["id"] for c in cases}) == 30
    assert {(c["layout"], c["mass_kg"]) for c in cases} == {
        (layout, mass) for layout in ("left", "right") for mass in (0.02, 0.05, 0.1)
    }
    assert all(c["frozen_fixture"]["accepted"] for c in cases)
    assert all(len(c["frozen_fixture"]["witness_hand"]) == 9 for c in cases)


def test_structural_gate_accepts_complete_consistent_evidence(report):
    assert benchmark.require_public_readiness(report)


def test_same_shape_altered_suite_cannot_self_certify(report):
    report["suite"] = copy.deepcopy(report["suite"])
    report["suite"]["contract"]["retention_rotation_rad"] = 1.0
    report["suite_sha256"] = hashlib.sha256(json.dumps(report["suite"], sort_keys=True).encode()).hexdigest()
    with pytest.raises(ValueError, match="canonical catalog"):
        benchmark.require_public_readiness(report)


@pytest.mark.parametrize("change", ["missing", "duplicate", "source", "fixture", "agent", "nonfinite"])
def test_incomplete_stale_or_mislabeled_evidence_rejected(report, change):
    if change == "missing":
        report["cases"].pop()
    elif change == "duplicate":
        report["cases"][-1] = copy.deepcopy(report["cases"][0])
    elif change == "source":
        report["source_hashes"] = {"test_manifest": "old"}
    elif change == "fixture":
        report["cases"][0]["model_signature"] = "changed"
    elif change == "agent":
        report["cases"][0]["selector"] = "llm_privileged"
    else:
        report["cases"][0]["contact_metrics"]["peak_contact_force"] = float("nan")
    with pytest.raises(ValueError):
        benchmark.require_public_readiness(report)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("post_execution_stable_s", 0),
        ("peak_base_displacement_m", 0),
        ("max_payload_rotation_rad", 0.101),
        ("max_payload_translation_m", 0.021),
        ("robot_failure", "retention_lost"),
        ("calls", []),
    ],
)
def test_task_failure_cannot_be_hidden_by_success_boolean(report, field, value):
    report["cases"][0][field] = value
    with pytest.raises(ValueError):
        benchmark.require_public_readiness(report)


@pytest.mark.parametrize(
    ("field", "value"), [("wheel_lift", True), ("wheel_support_evaluated", False), ("peak_contact_force", 30.01)]
)
def test_safety_failure_rejected(report, field, value):
    report["cases"][0]["contact_metrics"][field] = value
    with pytest.raises(ValueError, match="force/wheel"):
        benchmark.require_public_readiness(report)


def test_motion_metrics_keep_failed_motion_and_measure_jerk():
    from emet.eval.mars_tamp_metrics import motion_metrics

    rotation = np.eye(3)[:, :2].ravel().tolist()
    trace = [
        {
            "time": float(i),
            "hand": [0.01, 0, 0, *rotation],
            "hand_goal": [0, 0, 0, *rotation],
            "joint_velocity": [float(i * i)] * 5,
            "commanded_joint_velocity": [0] * 5,
            "failure": "diagnostic" if i == 2 else None,
        }
        for i in range(3)
    ]
    metrics = motion_metrics(trace)
    assert metrics["samples"] == 3
    assert metrics["hand_position_rms_m"] == pytest.approx(0.01)
    assert metrics["hand_rotation_max_rad"] == 0
    assert metrics["arm_acceleration_max_rad_s2"] == 3
    assert metrics["arm_jerk_max_rad_s3"] == 2


def test_summary_keeps_admission_planning_and_execution_denominators():
    from emet.eval.mars_tamp_metrics import summarize

    rows = [
        {"case": {"layout": "left", "mass_kg": 0.02}, "fixture_admission": {"accepted": True}, "success": False},
        {
            "case": {"layout": "left", "mass_kg": 0.02},
            "fixture_admission": {"accepted": False, "reason": "no_stable_grasp_setup"},
            "success": False,
        },
    ]
    result = summarize(rows)
    assert result["strata"]["left:20g"] == {
        "cases": 2,
        "admitted": 1,
        "planned": 0,
        "execution_started": 0,
        "successes": 0,
        "end_to_end_success_rate": 0,
    }
    assert result["failure_stages"] == {"planning:no_admitted_plan": 1, "fixture:no_stable_grasp_setup": 1}
