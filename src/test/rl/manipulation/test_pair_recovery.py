# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Recovery must preserve exact numerics and explicitly bind both source IDs."""

import copy
import json
import runpy
from pathlib import Path

import pytest
import torch

from emet.rl.manipulation import pair_recovery as recovery
from emet.rl.manipulation.whole_body_training import verify_initial_pair


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    def write(path, value):
        path.write_text(json.dumps(value))
        return str(path)

    original = tmp_path / "original.py"
    original.write_text("# frozen original optimizer\n")
    script = tmp_path / "replay.py"
    script.write_text("# recorded offline replay\n")
    old_sources = dict.fromkeys(recovery.PROTECTED, "same") | {
        "@runtime": "unchanged",
        "rl/manipulation/whole_body_training.py": "before",
        "rl/manipulation/multi_critic_ppo.py": recovery.sha(original),
    }
    new_sources = old_sources | {"rl/manipulation/whole_body_training.py": "after"}
    reference = dict.fromkeys(
        (
            "suite_sha256",
            "initial_actor_weights_sha256",
            "initial_policy_state_sha256",
            "initial_optimizer_sha256",
            "rollout_rng_sha256",
            "initial_numpy_rng_sha256",
        ),
        "same",
    ) | {
        "source_hashes": old_sources,
        "seed": 90,
        "requested_decisions": 4096,
        "basic_cases_sha256": "basic",
        "training_family_scope": "basic_general_only",
        "privileged_critic": True,
        "ppo": {"advantage_normalization": "per_head", "gamma": 0.9975},
        "first_rollout": {"arrays_sha256": "all-arrays"},
    }
    current = copy.deepcopy(reference)
    current["source_hashes"] = new_sources
    current["ppo"]["advantage_normalization"] = "shared"
    proof = {
        "format": "mars-c1-source-recovery-proof-v1",
        "replay_script": str(script),
        "reference_ppo_source": str(original),
        "current_sources_digest": recovery.canonical(new_sources),
    }
    rows = [
        {"id": str(i), "mastered": True, "admission": {"initial_state_sha256": str(i), "reference_sha256": str(i)}}
        for i in range(164)
    ]
    for label, record in (("reference", reference), ("current", current)):
        teacher = tmp_path / (label + "-teacher.json")
        write(teacher, {"source_hashes": record["source_hashes"], "cases": rows})
        preflight = tmp_path / (label + "-preflight.json")
        write(
            preflight,
            {
                "source_hashes": record["source_hashes"],
                "ready_for_experimental_training": True,
                "general_teacher_sha256": recovery.sha(teacher),
            },
        )
        record["preflight_sha256"] = recovery.sha(preflight)
        proof[label] = {"teacher": str(teacher), "preflight": str(preflight)}
    ref_path = tmp_path / "reference-run.json"
    write(ref_path, reference)
    ref_sha = recovery.sha(ref_path)
    recipe = {
        "format": "mars-c1-source-recovery-v1",
        "reference_run_sha256": ref_sha,
        "reference_run_digest": recovery.canonical(reference),
        "basic_cases_sha256": "basic",
        "replay_script_sha256": recovery.sha(script),
        "source_changes": {
            "rl/manipulation/whole_body_training.py": {
                "before": "before",
                "after": "after",
                "reason": "recorded settings repair",
            }
        },
    }
    recipe_path = tmp_path / "recipe.json"
    write(recipe_path, recipe)
    monkeypatch.setattr(recovery, "RECIPE", recipe_path)
    proof["recipe_sha256"] = recovery.sha(recipe_path)
    proof["reference_run_sha256"] = ref_sha
    proof["optimizer_replay"] = {"first_rollout": reference["first_rollout"]}
    state = {"policy": {"weight": torch.ones(1)}, "optimizer": {"state": {0: {"step": torch.tensor(1.0)}}}}
    for label in ("reference", "current"):
        path = tmp_path / (label + ".pt")
        torch.save(state, path)
        proof["optimizer_replay"][label] = {"path": str(path), "sha256": recovery.sha(path)}
    proof_path = tmp_path / "proof.json"
    write(proof_path, proof)
    return reference, current, proof_path, ref_sha


def test_recovery_preserves_source_identities_and_still_requires_exact_numeric_pair(bridge):
    reference, current, proof, ref_sha = bridge
    before = copy.deepcopy((reference, current))
    verify_initial_pair(reference, current, recovery_proof=proof, reference_sha256=ref_sha)
    assert (reference, current) == before
    with pytest.raises(ValueError, match="paired initial"):
        verify_initial_pair(reference, current)  # No implicit bridge.
    for key in ("initial_policy_state_sha256", "first_rollout", "initial_optimizer_sha256"):
        changed = copy.deepcopy(current)
        changed[key] = "different"
        with pytest.raises(ValueError, match="paired initial"):
            verify_initial_pair(reference, changed, recovery_proof=proof, reference_sha256=ref_sha)


@pytest.mark.parametrize("change", ["runtime", "scope", "mode", "source", "proof_source", "optimizer", "teacher"])
def test_recovery_rejects_unrelated_changes_even_with_consistent_artifact_checksums(bridge, change):
    reference, current, path, ref_sha = bridge
    proof = recovery.read(path)
    if change == "runtime":
        current["source_hashes"]["@runtime"] = "different"
    elif change == "scope":
        current["training_family_scope"] = "general_tamp_50_50_ticks"
    elif change == "mode":
        current["ppo"]["advantage_normalization"] = "per_head"
    elif change == "source":
        current["source_hashes"][recovery.PROTECTED[0]] = "changed actor"
    elif change == "proof_source":
        proof["current_sources_digest"] = "stale"
    elif change == "optimizer":
        artifact = proof["optimizer_replay"]["current"]
        state = torch.load(artifact["path"], weights_only=True)
        state["policy"]["weight"] += 1
        torch.save(state, artifact["path"])
        artifact["sha256"] = recovery.sha(artifact["path"])
    else:
        paths = proof["current"]
        teacher = recovery.read(paths["teacher"])
        teacher["cases"][0]["admission"]["reference_sha256"] = "different geometry"
        from pathlib import Path

        Path(paths["teacher"]).write_text(json.dumps(teacher))
        preflight = recovery.read(paths["preflight"])
        preflight["general_teacher_sha256"] = recovery.sha(paths["teacher"])
        Path(paths["preflight"]).write_text(json.dumps(preflight))
        current["preflight_sha256"] = recovery.sha(paths["preflight"])
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError):
        verify_initial_pair(reference, current, recovery_proof=path, reference_sha256=ref_sha)


def test_protected_source_cannot_be_whitelisted():
    recipe = {"source_changes": {recovery.PROTECTED[0]: {"before": "a", "after": "b", "reason": "no"}}}
    with pytest.raises(ValueError, match="cannot change actor"):
        recovery.source_equivalence({recovery.PROTECTED[0]: "a"}, {recovery.PROTECTED[0]: "b"}, recipe)


def test_actual_original_gae_remains_equal_after_adding_only_contract_docstring():
    definition = runpy.run_path(str(recovery.ROOT / "scripts/prove_mars_c1_recovery.py"))["definition"]
    original = Path(__file__).with_name("c1_original_gae.py").read_text()
    current = (recovery.ROOT / "src/emet/rl/manipulation/multi_critic_ppo.py").read_text()
    assert definition(original, "grouped_gae") == definition(current, "grouped_gae")
    changed = current.replace("delta = rewards[t]", "delta = 2 * rewards[t]")
    assert definition(original, "grouped_gae") != definition(changed, "grouped_gae")
