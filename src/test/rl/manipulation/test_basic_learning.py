# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Basic diagnostics select existing cases without changing goals or gates."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from emet.rl.manipulation.basic_learning import load_basic_learning
from emet.rl.manipulation.carry_benchmark import digest
from emet.rl.manipulation.mastery import load_suite, select_cases

SPEC = Path("configs/benchmarks/mars_whole_body_basic_learning_v1.json")


def test_basic_sets_are_exact_subsets_and_do_not_mutate_parent():
    parent = load_suite("configs/benchmarks/mars_tracking_mastery_supported_v1.json")
    before = copy.deepcopy(parent)
    spec, training, evaluation = load_basic_learning(SPEC, parent)
    originals = {c["id"]: c for c in select_cases(parent, ("base", "arm", "coupled"), "all", population="admission")}
    assert digest(parent) == digest(before)
    assert training["gate"] == evaluation["gate"] == parent["gate"]
    assert training["reference"] == evaluation["reference"] == parent["reference"]
    assert training["config"] == evaluation["config"] == parent["config"]
    assert len(training["cases"]) == 3 and len(evaluation["cases"]) == 12
    for suite, key in ((training, "train_case_ids"), (evaluation, "evaluation_case_ids")):
        assert suite["cases"] == [originals[cid] for cid in spec[key]]
    assert training["training_sampling"] == "control_steps"
    assert "extra_training_cases" not in training


@pytest.mark.parametrize(
    "change", ["duplicate", "unknown", "validation", "contact_train", "missing_eval", "parent_hash"]
)
def test_invalid_basic_specifications_are_rejected(tmp_path, change):
    spec = json.loads(SPEC.read_text())
    if change == "duplicate":
        spec["train_case_ids"].append(spec["train_case_ids"][0])
    elif change == "unknown":
        spec["train_case_ids"][1] = "unknown"
    elif change == "validation":
        spec["train_case_ids"][1] = "validation:arm:pitch_up"
    elif change == "contact_train":
        spec["train_case_ids"][1] = "train:arm:supported_A_up"
    elif change == "missing_eval":
        spec["evaluation_case_ids"].remove(spec["train_case_ids"][0])
    else:
        spec["parent_suite_file_sha256"] = "wrong"
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    parent = load_suite(spec["parent_suite"])
    with pytest.raises(ValueError):
        load_basic_learning(path, parent)


@pytest.mark.sim
def test_basic_environment_has_no_tamp_and_one_tick_per_decision(monkeypatch):
    import emet.rl.manipulation.tamp_primitive_env as tamp
    from emet.rl.manipulation.whole_body_training import make_environment

    def forbidden(*args, **kwargs):
        raise AssertionError("basic diagnostic must not allocate TAMP")

    monkeypatch.setattr(tamp, "TampPrimitiveEnv", forbidden)
    env = make_environment("configs/benchmarks/mars_whole_body_v2_development.json", basic_cases=SPEC)
    try:
        obs, info = env.reset(seed=90)
        assert env.names == ("general",) and info["training_family"] == "general"
        assert env.observation_space.contains(obs)
        _, _, _, _, info = env.step(np.zeros(8))
        assert info["elapsed_control_ticks"] == 1
        assert info["mastery"]["case_id"] in json.loads(SPEC.read_text())["train_case_ids"]
    finally:
        env.close()


def test_training_persists_paired_initial_state_and_first_rollout(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    import emet.rl.manipulation.whole_body_training as training
    import emet.rl.regression_gate as gate
    from emet.rl.manipulation.env import ManipulationConfig
    from emet.rl.manipulation.observation_layout import history_layout
    from emet.rl.manipulation.profiles import MARS

    class Env:
        def __init__(self):
            self.ticks = 0
            native = SimpleNamespace(
                config=ManipulationConfig(
                    robot="innate_mars", action_mode="direct_velocity", retention_context=True, trajectory_preview=True
                ),
                fingerprint="toy-model",
            )
            self.environments = {"general": SimpleNamespace(unwrapped=native)}

        def reset(self, **kwargs):
            fields, width = history_layout(MARS, retention_context=True)
            h = np.zeros((20, width), np.float32)
            h[:, fields["base_mask"]] = 1
            return {
                "history": h,
                "preview": np.zeros((7, 18), np.float32),
                "motion": np.zeros(9, np.float32),
                "critic": np.zeros(34, np.float32),
            }, {}

        def step(self, action):
            self.ticks += 1
            obs, _ = self.reset()
            return (
                obs,
                0.0,
                False,
                False,
                {
                    "elapsed_control_ticks": 1,
                    "reward_groups": np.array([0.1, -0.01, 0.0], np.float32),
                    "mastery": {"case_id": "toy"},
                },
            )

        def ledger(self):
            return {"general": {"control_ticks": self.ticks, "episodes": 1, "requested_fraction": 1.0}}

        def close(self):
            pass

    monkeypatch.setattr(training, "make_environment", lambda *args, **kwargs: Env())
    monkeypatch.setattr(training, "require_preflight", lambda *args: None)
    monkeypatch.setattr(gate, "verify_regressions", lambda *args: {"test_ids": ["fixture"]})
    receipt, preflight = tmp_path / "receipt.json", tmp_path / "preflight.json"
    receipt.write_text("{}")
    preflight.write_text("{}")
    records = []
    for mode in ("per_head", "shared"):
        target = tmp_path / mode
        args = SimpleNamespace(
            regression_receipt=receipt,
            preflight=preflight,
            suite=Path("configs/benchmarks/mars_whole_body_v2_development.json"),
            seed=90,
            output=target,
            learning_rate=3e-4,
            epochs=1,
            batch_size=2,
            advantage_normalization=mode,
            no_privileged_critic=False,
            basic_cases=None,
            wandb_mode="disabled",
            steps=2,
            rollout_steps=2,
        )
        training.train(args)
        record = json.loads((target / "run.json").read_text())
        assert record["ppo"]["advantage_normalization"] == mode
        initial = torch.load(target / "initial/training.pt", weights_only=True)
        assert {"policy", "optimizer", "torch_rng", "numpy_rng"} == set(initial)
        assert initial["optimizer"]["state"] == {}
        saved = torch.load(target / "training.pt", weights_only=True)
        assert saved["run"]["ppo"]["advantage_normalization"] == mode
        with np.load(target / "first-rollout.npz", allow_pickle=False) as rollout:
            assert rollout["case_ids"].tolist() == ["toy", "toy"]
        records.append(record)
    for key in (
        "initial_actor_weights_sha256",
        "initial_policy_state_sha256",
        "initial_optimizer_sha256",
        "rollout_rng_sha256",
        "initial_numpy_rng_sha256",
        "first_rollout",
    ):
        assert records[0][key] == records[1][key]


def test_pair_verification_rejects_state_data_or_optimizer_changes_before_update():
    from emet.rl.manipulation.whole_body_training import verify_initial_pair

    keys = (
        "source_hashes",
        "seed",
        "suite_sha256",
        "preflight_sha256",
        "basic_cases_sha256",
        "requested_decisions",
        "initial_actor_weights_sha256",
        "initial_policy_state_sha256",
        "initial_optimizer_sha256",
        "rollout_rng_sha256",
        "initial_numpy_rng_sha256",
        "first_rollout",
    )
    first = dict.fromkeys(keys, "same")
    first["ppo"] = {"learning_rate": 0.0003, "advantage_normalization": "per_head"}
    second = copy.deepcopy(first)
    second["ppo"]["advantage_normalization"] = "shared"
    verify_initial_pair(first, second)
    for key in keys:
        changed = copy.deepcopy(second)
        changed[key] = "different"
        with pytest.raises(ValueError, match="paired"):
            verify_initial_pair(first, changed)
    second["ppo"]["learning_rate"] = 0.1
    with pytest.raises(ValueError, match="PPO settings"):
        verify_initial_pair(first, second)
