# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unloaded-stage admission cannot borrow a different boundary or exception."""

import copy

import pytest

from emet.rl.manipulation.primitive_admission import unloaded_stage_readiness


def success_row(primitive):
    return {
        "id": "case",
        "primitive": primitive,
        "success": True,
        "code": primitive + "_complete",
        "boundary_reached": True,
        "terminal_pose_in_contract": True,
        "gripper_open": True,
        "stop_confirmed": True,
        "initialization_failed": False,
        "full_pick_place_success": False,
        "decisions": 20,
        "commands": [{"phase": "pregrasp"}] + ([{"phase": "grasp"}] if primitive == "grasp_pose" else []),
        "inference_counts": {"policy_calls": 0, "reference_controller_calls": 20},
        "safety_flags": dict.fromkeys(("wheel_lift", "invalid_physics", "unsafe_contact", "retention_lost"), False),
    }


@pytest.mark.parametrize("primitive", ["pregrasp", "grasp_pose"])
def test_separate_readiness_requires_its_actual_boundary(primitive):
    row = success_row(primitive)
    assert unloaded_stage_readiness([row], [{"id": "case"}], primitive)
    other = "grasp_pose" if primitive == "pregrasp" else "pregrasp"
    assert not unloaded_stage_readiness([row], [{"id": "case"}], other)


def test_pregrasp_readiness_rejects_execution_beyond_its_boundary():
    row = success_row("pregrasp")
    row["commands"].append({"phase": "grasp"})
    assert not unloaded_stage_readiness([row], [{"id": "case"}], "pregrasp")


@pytest.mark.parametrize(
    "change",
    [
        {"success": False, "code": "object_changed_replan"},
        {"code": "pregrasp_complete"},
        {"boundary_reached": False},
        {"terminal_pose_in_contract": False},
        {"gripper_open": False},
        {"stop_confirmed": None},
        {"initialization_failed": True},
        {"full_pick_place_success": True},
        {"decisions": 0},
        {"commands": []},
        {"commands": [{"gripper": "close"}]},
        {"commands": [{"phase": "lift"}]},
        {"safety_flags": {}},
        {"safety_flags": dict.fromkeys(("wheel_lift", "invalid_physics", "unsafe_contact", "retention_lost"), 0)},
        {"inference_counts": {"policy_calls": 20, "reference_controller_calls": 0}},
    ],
)
def test_rejects_failed_or_misidentified_stage_evidence(change):
    row = success_row("grasp_pose") | copy.deepcopy(change)
    assert not unloaded_stage_readiness([row], [{"id": "case"}], "grasp_pose")


def test_no_missing_duplicate_or_empty_case_cohort():
    row = success_row("grasp_pose")
    assert not unloaded_stage_readiness([], [{"id": "case"}], "grasp_pose")
    assert not unloaded_stage_readiness([row, row], [{"id": "case"}, {"id": "case"}], "grasp_pose")
    assert not unloaded_stage_readiness([], [], "grasp_pose")


def test_old_pregrasp_readiness_cannot_admit_a_grasp_pose_row():
    from emet.rl.manipulation.whole_body_training import tamp_training_readiness

    row = success_row("grasp_pose")
    suite = {"train_cases": [{"id": "case"}], "audit_cases": [], "teacher_expected_failures": {}}
    assert not tamp_training_readiness([row], suite)


@pytest.mark.parametrize("source_drift", [False, True])
def test_bounded_runner_retains_failures_and_rejects_source_drift(tmp_path, monkeypatch, source_drift):
    import json
    import runpy
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    run = runpy.run_path(str(root / "scripts/validate_mars_unloaded_primitives.py"))["run"]
    globals_ = run.__globals__
    cases = [{"id": str(i)} for i in range(6)]
    monkeypatch.setitem(
        globals_, "load_development", lambda: ({"train_cases": cases[:4], "audit_cases": cases[4:]}, {})
    )
    calls = []

    def rollout(actual, *, primitive, wheel_feedback_current_velocity, trace_dir):
        assert actual == cases and wheel_feedback_current_velocity is True
        assert trace_dir.name == primitive
        calls.append(primitive)
        rows = [success_row(primitive) | {"id": c["id"]} for c in cases]
        rows[2].update(success=False, code="object_changed_replan", boundary_reached=False)
        return rows

    monkeypatch.setitem(globals_, "primitive_rollout", rollout)
    monkeypatch.setitem(
        globals_, "source_state", lambda _: {"source": "drift" if source_drift and len(calls) == 2 else "frozen"}
    )
    output = tmp_path / "new-admission"
    if source_drift:
        with pytest.raises(RuntimeError, match="source changed"):
            run(output)
    else:
        run(output)
    report = json.loads((output / "results.json").read_text())
    assert report["complete"] is not source_drift
    assert report["new_ppo_steps"] == 0 and not report["ready_for_training"]
    assert calls == ["pregrasp", "grasp_pose"]
    for primitive in calls:
        stage = report["stages"][primitive]
        assert stage["attempts"] == 6 and stage["successes"] == 5 and not stage["stage_teacher_ready"]
        assert stage["rows"][2]["code"] == "object_changed_replan"
    with pytest.raises(FileExistsError):
        run(output)
