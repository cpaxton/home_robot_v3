# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Prepared coverage and audit identities are not teacher admission."""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from emet.rl.manipulation.directional_curriculum import SPEC, load_directional_curriculum


def test_prepared_suites_preserve_contract_and_separate_training_audit():
    before = Path("configs/benchmarks/mars_tracking_mastery_supported_v1.json").read_bytes()
    spec, control, training, diagnostic, audit = load_directional_curriculum()
    assert [len(s["cases"]) for s in (control, training, diagnostic, audit)] == [3, 17, 12, 24]
    assert Counter(c["stage"] for c in training["cases"]) == {"base": 5, "arm": 10, "coupled": 2}
    assert Counter(c["stage"] for c in audit["cases"]) == {"base": 8, "arm": 8, "coupled": 8}
    assert training["training_sampling"] == "control_steps"
    assert training["training_stage_weights"] == {"base": 0.3, "arm": 0.5, "coupled": 0.2}
    assert control["training_stage_weights"] == {"base": 0.5, "arm": 0.5}
    for suite in (control, training, diagnostic, audit):
        assert suite["config"]["wheel_feedback_current_velocity"] is True
        assert suite["gate"] == control["gate"] and suite["reference"] == control["reference"]
        assert not any(k in suite for k in ["extra_training_cases", "training_population", "arm_training_scales"])
    assert [c["id"] for c in diagnostic["cases"]] == spec["evaluation_case_ids"]
    assert not {c["reset_seed"] for c in training["cases"]} & {c["reset_seed"] for c in audit["cases"]}
    assert Path("configs/benchmarks/mars_tracking_mastery_supported_v1.json").read_bytes() == before
    training["cases"][0]["arm_delta"][0] = 99
    assert control["cases"][0]["arm_delta"][0] == 0


@pytest.mark.parametrize(
    "change",
    [
        "parent_hash",
        "control_hash",
        "config",
        "weights",
        "duplicate_train",
        "missing_axis",
        "missing_base",
        "unbalanced_coupled",
        "diagnostic",
        "audit_count",
        "audit_seed",
        "audit_id",
        "audit_bound",
        "audit_nan",
        "audit_bool",
        "audit_single_axis",
        "audit_parent_motion",
        "audit_posture",
        "audit_unknown",
    ],
)
def test_rejects_changed_contract_incomplete_coverage_and_leaking_audits(tmp_path, change):
    d = json.loads(SPEC.read_text())
    if change == "parent_hash":
        d["parent_suite_file_sha256"] = "wrong"
    elif change == "control_hash":
        d["control_spec_file_sha256"] = "wrong"
    elif change == "config":
        d["config_overrides"]["base_xy_tolerance"] = 0.1
    elif change == "weights":
        d["training_stage_weights"]["base"] = 0.6
    elif change == "duplicate_train":
        d["train_case_ids"][1] = d["train_case_ids"][0]
    elif change == "missing_axis":
        d["train_case_ids"][5] = "train:arm:translate_up"
    elif change == "missing_base":
        d["train_case_ids"][1] = "train:base:hold@posture:forward"
    elif change == "unbalanced_coupled":
        d["train_case_ids"][-1] = "train:coupled:forward_turn@amplitude:0.25"
    elif change == "diagnostic":
        d["evaluation_case_ids"].pop()
    elif change == "audit_count":
        d["audit_cases"].pop()
    elif change == "audit_seed":
        d["audit_cases"][0]["reset_seed"] = 34000000
    elif change == "audit_id":
        d["audit_cases"][1]["id"] = d["audit_cases"][0]["id"]
    elif change == "audit_bound":
        d["audit_cases"][0]["travel_m"] = 0.051
    elif change == "audit_nan":
        d["audit_cases"][0]["turn_rad"] = float("nan")
    elif change == "audit_bool":
        d["audit_cases"][0]["travel_m"] = True
    elif change == "audit_single_axis":
        d["audit_cases"][8]["arm_delta"] = [0.02, 0, 0, 0, 0]
    elif change == "audit_parent_motion":
        d["audit_cases"][8]["arm_delta"] = [0, -0.01, 0.03, -0.02, 0]
    elif change == "audit_posture":
        d["audit_cases"][0]["arm_start"] = [0] * 5
    elif change == "audit_unknown":
        d["audit_cases"][0]["supported_fixture"] = True
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        load_directional_curriculum(p)


@pytest.mark.parametrize("optimized", [False, True])
def test_parent_mutation_fails_closed_even_under_python_optimization(optimized):
    # Mutate after the initial hashes pass, exercising the final invariant.
    code = '''
from emet.rl.manipulation import directional_curriculum as curriculum

original = curriculum.load_basic_learning
def mutate(path, parent):
    result = original(path, parent)
    parent["scope"] += " changed"
    return result

curriculum.load_basic_learning = mutate
try:
    curriculum.load_directional_curriculum()
except ValueError as exc:
    if str(exc) != "directional curriculum parent changed while loading":
        raise
else:
    raise RuntimeError("mutated parent was accepted")
'''
    subprocess.run([sys.executable, *(["-O"] if optimized else []), "-c", code], check=True)
