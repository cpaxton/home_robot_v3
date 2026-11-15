# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Prepared general-task coverage; loading never admits a reference or trains."""

import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from emet.rl.manipulation.basic_learning import load_basic_learning
from emet.rl.manipulation.carry_benchmark import digest
from emet.rl.manipulation.mastery import load_suite, select_cases

SPEC = Path("configs/benchmarks/mars_directional_curriculum_v1.json")
STAGES = ("base", "arm", "coupled")
KEYS = {
    "format",
    "scope",
    "parent_suite",
    "parent_suite_file_sha256",
    "control_spec",
    "control_spec_file_sha256",
    "config_overrides",
    "training_stage_weights",
    "train_case_ids",
    "evaluation_case_ids",
    "audit_cases",
}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _motion(case):
    return (case["travel_m"], case["turn_rad"], *case["arm_delta"])


def load_directional_curriculum(path=SPEC):
    """Return manifest/control/training/diagnostic/audit with unchanged gates.

    Like existing benchmark CLIs, paths are repository-root relative. Runtime
    admission must independently cover these cases and the selected plant.
    """
    spec = json.loads(Path(path).read_text())
    if (
        not isinstance(spec, dict)
        or set(spec) != KEYS
        or spec["format"] != "mars-directional-curriculum-v1"
        or not isinstance(spec["scope"], str)
        or not spec["scope"].strip()
        or spec["config_overrides"] != {"wheel_feedback_current_velocity": True}
        or type(spec["config_overrides"]["wheel_feedback_current_velocity"]) is not bool
        or spec["training_stage_weights"] != {"base": 0.3, "arm": 0.5, "coupled": 0.2}
    ):
        raise ValueError("unsupported directional curriculum specification")
    for name in ("parent_suite", "control_spec"):
        if hashlib.sha256(Path(spec[name]).read_bytes()).hexdigest() != spec[name + "_file_sha256"]:
            raise ValueError("directional curriculum parent/control file changed")
    parent = load_suite(spec["parent_suite"])
    control_spec, control, diagnostic = load_basic_learning(spec["control_spec"], parent)
    cases = select_cases(parent, STAGES, "all", population="admission")
    originals = {c["id"]: c for c in cases}
    ids = spec["train_case_ids"]
    if (
        not isinstance(ids, list)
        or len(ids) != 17
        or any(not isinstance(i, str) for i in ids)
        or len(set(ids)) != 17
        or not set(ids).issubset(originals)
        or spec["evaluation_case_ids"] != control_spec["evaluation_case_ids"]
    ):
        raise ValueError("seventeen unique existing train cases and frozen diagnostic IDs required")
    training = [copy.deepcopy(originals[i]) for i in ids]
    if Counter(c["stage"] for c in training) != {"base": 5, "arm": 10, "coupled": 2} or any(
        c["split"] != "train" or c.get("supported_fixture") for c in training
    ):
        raise ValueError("directional training requires five base, ten free-arm and two coupled cases")
    basis = set()
    for case in (c for c in training if c["stage"] == "arm"):
        axes = [i for i, value in enumerate(case["arm_delta"]) if value != 0]
        if len(axes) != 1:
            raise ValueError("arm training requires both signs of each single-joint direction")
        axis = axes[0]
        basis.add((axis, 1 if case["arm_delta"][axis] > 0 else -1))
    if basis != {(i, sign) for i in range(5) for sign in (-1, 1)}:
        raise ValueError("arm training requires both signs of each single-joint direction")
    roles = set()
    for case in (c for c in training if c["stage"] == "base"):
        travel, turn = case["travel_m"], case["turn_rad"]
        if travel and turn:
            raise ValueError("base training requires hold and signed translation/yaw primitives")
        roles.add(
            "forward"
            if travel > 0
            else "reverse"
            if travel < 0
            else "left"
            if turn > 0
            else "right"
            if turn < 0
            else "hold"
        )
    if roles != {"hold", "forward", "reverse", "left", "right"}:
        raise ValueError("base training requires hold and signed translation/yaw primitives")
    paired = [_motion(c) for c in training if c["stage"] == "coupled"]
    if any(abs(a + b) > 1e-12 for a, b in zip(*paired, strict=True)):
        raise ValueError("coupled training requires opposing paired directions")

    audit = spec["audit_cases"]
    if not isinstance(audit, list) or len(audit) != 24:
        raise ValueError("exactly twenty-four audit cases required")
    seen_ids, seen_seeds, seen_motion = set(originals), {c["reset_seed"] for c in cases}, set()
    parent_motion = {(c["stage"], _motion(c)) for c in cases}
    for case in audit:
        required = {"id", "split", "stage", "reset_seed", "travel_m", "turn_rad", "arm_delta"}
        if (
            not isinstance(case, dict)
            or not required.issubset(case)
            or set(case) - required - {"arm_start"}
            or case["stage"] not in STAGES
            or case["split"] != "validation"
            or not isinstance(case["id"], str)
            or not case["id"].startswith("audit:" + case["stage"] + ":")
            or case["id"] in seen_ids
            or type(case["reset_seed"]) is not int
            or not 0 <= case["reset_seed"] < 2**31
            or case["reset_seed"] in seen_seeds
            or not isinstance(case["arm_delta"], list)
            or len(case["arm_delta"]) != 5
        ):
            raise ValueError("unique disjoint audit identities and known case fields required")
        motion = _motion(case)
        if any(not _finite(v) or abs(v) > limit for v, limit in zip(motion, (0.05, 0.1, *[0.2] * 5), strict=True)):
            raise ValueError("audit motion exceeds finite nearby bounds")
        nonzero = sum(v != 0 for v in case["arm_delta"])
        if (
            case["stage"] == "base"
            and (nonzero or not case["travel_m"] or not case["turn_rad"])
            or case["stage"] == "arm"
            and (case["travel_m"] or case["turn_rad"] or nonzero < 2)
            or case["stage"] == "coupled"
            and (not case["travel_m"] or not case["turn_rad"] or nonzero < 2)
            or (case["stage"], motion) in parent_motion | seen_motion
        ):
            raise ValueError("audit requires fresh combined directions without parent motion duplicates")
        if "arm_start" in case and (
            case["stage"] != "arm"
            or not isinstance(case["arm_start"], list)
            or len(case["arm_start"]) != 5
            or any(not _finite(v) or abs(v) > 1 for v in case["arm_start"])
        ):
            raise ValueError("audit arm starts require finite free-arm postures")
        seen_ids.add(case["id"])
        seen_seeds.add(case["reset_seed"])
        seen_motion.add((case["stage"], motion))
    if Counter(c["stage"] for c in audit) != dict.fromkeys(STAGES, 8):
        raise ValueError("eight audit cases per general stage required")

    def subset(rows, *, train=False):
        result = copy.deepcopy(diagnostic)
        result["cases"] = copy.deepcopy(rows)
        result["scope"] = spec["scope"]
        if train:
            result["training_stage_weights"] = spec["training_stage_weights"].copy()
            result["training_sampling"] = "control_steps"
        return result

    outputs = [control, subset(training, train=True), diagnostic, subset(audit)]
    for suite in outputs:
        suite["config"] = {**suite["config"], **spec["config_overrides"]}
        if suite["reference"] != parent["reference"] or suite["gate"] != parent["gate"]:
            raise ValueError("directional curriculum cannot change references or acceptance")
    assert digest(parent) == digest(json.loads(Path(spec["parent_suite"]).read_text()))
    return spec, *outputs
