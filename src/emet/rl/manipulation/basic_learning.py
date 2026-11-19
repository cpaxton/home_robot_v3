# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Explicit small-case diagnostics; frozen acceptance suites remain unchanged."""

import copy
import hashlib
import json
from pathlib import Path

from emet.rl.manipulation.carry_benchmark import digest
from emet.rl.manipulation.mastery import select_cases


def load_basic_learning(path, parent_suite):
    spec = json.loads(Path(path).read_text())
    expected = {
        "format",
        "scope",
        "parent_suite",
        "parent_suite_file_sha256",
        "training_stage_weights",
        "train_case_ids",
        "evaluation_case_ids",
    }
    if (
        set(spec) != expected
        or spec["format"] != "mars-whole-body-basic-learning-v1"
        or not isinstance(spec["scope"], str)
        or not spec["scope"].strip()
        or spec["training_stage_weights"] != {"base": 0.5, "arm": 0.5}
    ):
        raise ValueError("unsupported basic-learning diagnostic specification")
    parent = Path(spec["parent_suite"])
    if hashlib.sha256(parent.read_bytes()).hexdigest() != spec["parent_suite_file_sha256"]:
        raise ValueError("basic-learning parent file changed")
    if digest(json.loads(parent.read_text())) != digest(parent_suite):
        raise ValueError("basic-learning parent does not match development curriculum")
    cases = {c["id"]: c for c in select_cases(parent_suite, ("base", "arm", "coupled"), "all", population="admission")}
    for field in ("train_case_ids", "evaluation_case_ids"):
        ids = spec[field]
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(i, str) for i in ids)
            or len(set(ids)) != len(ids)
            or not set(ids).issubset(cases)
        ):
            raise ValueError("unique existing basic-learning case IDs required")
    train = [cases[i] for i in spec["train_case_ids"]]
    if (
        not set(spec["train_case_ids"]).issubset(spec["evaluation_case_ids"])
        or {c["stage"] for c in train} != {"base", "arm"}
        or any(c["split"] != "train" or c.get("supported_fixture", False) for c in train)
    ):
        raise ValueError("basic training requires free-space train cases and independent evaluation coverage")

    def subset(ids, *, training):
        result = copy.deepcopy(parent_suite)
        for key in (
            "extra_training_cases",
            "training_population",
            "training_stage_weights",
            "training_sampling",
            "training_primitive_weights",
            "arm_training_scales",
            "coupled_training_scales",
        ):
            result.pop(key, None)
        result["cases"] = [copy.deepcopy(cases[i]) for i in ids]
        result["scope"] = spec["scope"]
        if training:
            result["training_stage_weights"] = spec["training_stage_weights"].copy()
            result["training_sampling"] = "control_steps"
        return result

    return spec, subset(spec["train_case_ids"], training=True), subset(spec["evaluation_case_ids"], training=False)
