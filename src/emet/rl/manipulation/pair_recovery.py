# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Narrow, explicitly recorded source bridge for the interrupted C1 comparison.

This does not admit a controller or reuse a regression/teacher receipt. Current
source admission remains mandatory; numeric initial/rollout checks still run.
"""

import gzip
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
RECIPE = ROOT / "configs/benchmarks/mars_whole_body_advantage_recovery_v1.json"
PROTECTED = (
    "rl/manipulation/whole_body_actor.py",
    "rl/manipulation/whole_body_observation.py",
    "rl/manipulation/whole_body_reward.py",
    "rl/manipulation/whole_body_env.py",
    "rl/manipulation/observation_layout.py",
    "rl/manipulation/tracking.py",
    "rl/manipulation/control.py",
    "rl/manipulation/profiles.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_equivalence(reference, current, recipe):
    """Require an exact, reviewed before/after manifest; never rewrite either."""
    expected = dict(reference)
    changes = recipe["source_changes"]
    if any(name in changes for name in PROTECTED):
        raise ValueError("C1 recovery cannot change actor, references or physical reward/control code")
    for name, change in changes.items():
        if expected.get(name) != change["before"] or not change.get("reason"):
            raise ValueError("C1 recovery source change does not match recorded reference")
        if change["after"] is None:
            expected.pop(name, None)
        else:
            expected[name] = change["after"]
    if expected != current:
        raise ValueError("unrecorded C1 recovery source/runtime/asset change")


def tensor_states_equal(first, second):
    """Compare complete policy/optimizer tensors, dtypes and scalar state."""
    if type(first) is not type(second):
        return False
    if isinstance(first, torch.Tensor):
        return first.dtype == second.dtype and first.shape == second.shape and torch.equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(tensor_states_equal(first[k], second[k]) for k in first)
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(tensor_states_equal(a, b) for a, b in zip(first, second, strict=True))
    return first == second


def verify_recovery(reference, current, proof_path, *, reference_sha256):
    recipe, proof = read(RECIPE), read(proof_path)
    if (
        recipe.get("format") != "mars-c1-source-recovery-v1"
        or proof.get("format") != "mars-c1-source-recovery-proof-v1"
        or proof.get("recipe_sha256") != sha(RECIPE)
        or recipe["reference_run_sha256"] != reference_sha256
        or recipe["reference_run_digest"] != canonical(reference)
        or proof.get("reference_run_sha256") != reference_sha256
    ):
        raise ValueError("C1 recovery recipe/proof does not bind the original control")
    for record, mode in ((reference, "per_head"), (current, "shared")):
        if (
            record.get("basic_cases_sha256") != recipe["basic_cases_sha256"]
            or record.get("seed") != 90
            or record.get("requested_decisions") != 4096
            or record.get("training_family_scope") != "basic_general_only"
            or record.get("privileged_critic") is not True
            or record["ppo"]["advantage_normalization"] != mode
        ):
            raise ValueError("source recovery is limited to the declared one-tick C1 pair")
    source_equivalence(reference["source_hashes"], current["source_hashes"], recipe)
    if proof.get("current_sources_digest") != canonical(current["source_hashes"]):
        raise ValueError("C1 proof belongs to different current sources")
    if (
        sha(proof["replay_script"]) != recipe["replay_script_sha256"]
        or sha(proof["reference_ppo_source"]) != reference["source_hashes"]["rl/manipulation/multi_critic_ppo.py"]
    ):
        raise ValueError("C1 optimizer replay code mismatch")

    # Verify both actual receipts' teacher artifacts, preserving their source IDs.
    teachers = []
    for label, record in (("reference", reference), ("current", current)):
        paths = proof[label]
        if sha(paths["preflight"]) != record["preflight_sha256"]:
            raise ValueError("C1 recovery teacher artifact mismatch")
        preflight = read(paths["preflight"])
        if (
            preflight["source_hashes"] != record["source_hashes"]
            or not preflight["ready_for_experimental_training"]
            or sha(paths["teacher"]) != preflight["general_teacher_sha256"]
        ):
            raise ValueError("C1 recovery teacher/source mismatch")
        teacher = read(paths["teacher"])
        if teacher["source_hashes"] != record["source_hashes"]:
            raise ValueError("C1 teacher report source mismatch")
        rows = {row["id"]: row for row in teacher["cases"]}
        if len(rows) != len(teacher["cases"]) or len(rows) != 164:
            raise ValueError("C1 recovery requires all 164 unique teacher cases")
        teachers.append(rows)
    if teachers[0].keys() != teachers[1].keys():
        raise ValueError("C1 recovery teacher cases changed")
    for case_id, old in teachers[0].items():
        new = teachers[1][case_id]
        for key in ("initial_state_sha256", "reference_sha256"):
            if old["admission"][key] != new["admission"][key]:
                raise ValueError("C1 recovery teacher physical identity changed")
        if not old["mastered"] or not new["mastered"]:
            raise ValueError("C1 recovery teacher no longer masters the original cases")

    replay = proof["optimizer_replay"]
    if replay["first_rollout"] != reference["first_rollout"]:
        raise ValueError("C1 optimizer replay uses different rollout data")
    states = []
    for label in ("reference", "current"):
        artifact = replay[label]
        if sha(artifact["path"]) != artifact["sha256"]:
            raise ValueError("C1 optimizer replay state artifact mismatch")
        states.append(torch.load(artifact["path"], map_location="cpu", weights_only=True))
    if (
        set(states[0]) != {"policy", "optimizer"}
        or not states[0]["policy"]
        or not states[0]["optimizer"].get("state")
        or not tensor_states_equal(*states)
    ):
        raise ValueError("C1 optimizer replay changed policy or optimizer numerics")
    return {"recipe_sha256": sha(RECIPE), "proof_sha256": sha(proof_path), "teacher_cases": 164}
