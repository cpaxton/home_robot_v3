"""Component transfer preserves functions; it is not runtime qualification."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from emet.rl.manipulation.arm_component import SCOPE, ArmOnlyActions, ArmPolicy
from emet.rl.manipulation.carry_benchmark import digest, source_hashes
from emet.rl.manipulation.component_transfer import transfer_arm_component
from emet.rl.manipulation.feedback_policy import FeedbackActorCriticPolicy, GoalFeedbackFeatures
from emet.rl.manipulation.mastery import MasteryEnv, load_suite, save_report
from emet.rl.manipulation.normalization import CommandVecNormalize
from emet.rl.manipulation.policy import ManipulationPolicy, export


@pytest.fixture
def component(tmp_path):
    torch.set_num_threads(1)
    suite = load_suite(
        Path(__file__).resolve().parents[4] / "configs/benchmarks/mars_tracking_mastery_hand_accuracy_v1.json"
    )
    source_base = MasteryEnv(suite, stages=["arm"], split="train", population="training")
    target_base = MasteryEnv(suite, stages=["arm"], split="train", population="training")
    source_env = CommandVecNormalize(
        DummyVecEnv([lambda: ArmOnlyActions(source_base)]), config=source_base.config, norm_reward=False
    )
    target_env = CommandVecNormalize(DummyVecEnv([lambda: target_base]), config=target_base.config, norm_reward=False)
    kwargs = {
        "features_extractor_class": GoalFeedbackFeatures,
        "features_extractor_kwargs": {
            "robot": "innate_mars",
            "retention_context": True,
            "orientation_scale": source_base.config.feedback_orientation_scale,
        },
        "net_arch": {"pi": [16, 16], "vf": [16, 16]},
    }
    try:
        source = PPO(
            ArmPolicy, source_env, device="cpu", n_steps=8, batch_size=8, n_epochs=1, seed=19, policy_kwargs=kwargs
        )
        source.learn(16)
        target = PPO(
            FeedbackActorCriticPolicy,
            target_env,
            device="cpu",
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            seed=90,
            policy_kwargs={**kwargs, "restoring_yaw_feedback": True},
        )
        directory = tmp_path / "producer" / "last"
        directory.mkdir(parents=True)
        source.save(directory / "actor.zip")
        source_env.save(directory / "normalizer.pkl")
        hashes = source_hashes()
        # Synthetic success rows test evidence binding only, never quality.
        save_report(
            directory / "evaluation.json.gz",
            {
                "format": SCOPE,
                "controller": "ppo",
                "complete": True,
                "source_hashes": hashes,
                "suite_sha256": digest(suite),
                "cases": [
                    {"id": c["id"], "admitted": True, "mastered": True} for c in suite["cases"] if c["stage"] == "arm"
                ],
            },
        )
        (directory.parent / "component.json").write_text(
            json.dumps(
                {
                    "scope": SCOPE,
                    "action_size": 5,
                    "production_admitted": False,
                    "steps": 16,
                    "source_hashes": hashes,
                    "suite": suite,
                    "env": asdict(source_base.config),
                }
            )
        )
        (directory.parent / "complete.json").write_text(json.dumps({"steps": 16}))
        evidence = tmp_path / "evidence.json"
        evidence.write_text(
            json.dumps(
                {
                    "scope": SCOPE,
                    "production_admitted": False,
                    "steps": 16,
                    "source_hashes": hashes,
                    "artifact_checksums": {
                        "last/" + p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()
                    },
                }
            )
        )
        yield source, source_env, target, target_env, target_base, directory, evidence
    finally:
        source_env.close()
        target_env.close()


def transfer(component):
    _, _, target, env, base, directory, evidence = component
    return transfer_arm_component(target, env, directory, evidence, base.config, base.fingerprint)


@pytest.mark.sim
def test_component_transfer_preserves_arm_critic_normalization_and_runtime_roundtrip(component, tmp_path):
    source, source_env, target, target_env, base, _, _ = component
    result = transfer(component)
    assert result["arm_mean_function_preserving"] and result["critic_function_preserving"]
    assert not result["runtime_admitted"] and not result["optimizer_restored"]
    for key, stats in source_env.obs_rms.items():
        np.testing.assert_array_equal(stats.mean, target_env.obs_rms[key].mean)
        np.testing.assert_array_equal(stats.var, target_env.obs_rms[key].var)
        assert stats.count == target_env.obs_rms[key].count
    obs = base.reset()[0]
    rng = np.random.default_rng(314)
    target_env.training = False
    path = tmp_path / "whole-body"
    export(target, target_env, path, base.config, base.fingerprint, steps=0)
    runtime = ManipulationPolicy(path)
    for _ in range(8):
        changed = {k: v + rng.normal(0, 0.001, v.shape).astype(v.dtype) for k, v in obs.items()}
        left, right = source_env.normalize_obs(changed), target_env.normalize_obs(changed)
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
        expected = source.predict(left, deterministic=True)[0]
        actual = target.predict(right, deterministic=True)[0]
        np.testing.assert_allclose(actual[2:7], expected, rtol=0, atol=1e-7)
        np.testing.assert_allclose(actual[[0, 1, 7]], 0, atol=1e-7)
        np.testing.assert_allclose(runtime(changed), actual, rtol=0, atol=1e-7)
        with torch.no_grad():
            old_value = source.policy.predict_values(source.policy.obs_to_tensor(left)[0])
            new_value = target.policy.predict_values(target.policy.obs_to_tensor(right)[0])
        torch.testing.assert_close(old_value, new_value, rtol=0, atol=1e-7)


@pytest.mark.sim
@pytest.mark.parametrize("name", ["actor.zip", "normalizer.pkl", "evaluation.json.gz"])
def test_component_transfer_rejects_checksum_before_deserializing(component, monkeypatch, name):
    directory = component[-2]
    with (directory / name).open("ab") as stream:
        stream.write(b"changed")

    def forbidden(*args, **kwargs):
        pytest.fail("unverified artifact was deserialized")

    monkeypatch.setattr(PPO, "load", forbidden)
    monkeypatch.setattr("emet.rl.manipulation.component_transfer.pickle.load", forbidden)
    with pytest.raises(ValueError, match="checksum"):
        transfer(component)


@pytest.mark.sim
@pytest.mark.parametrize("change", ["steps", "configuration", "control_sources"])
def test_component_transfer_rejects_changed_producer_contract(component, change):
    directory = component[-2]
    path = directory.parent / "component.json"
    value = json.loads(path.read_text())
    if change == "steps":
        value["steps"] += 1
    elif change == "configuration":
        value["env"]["wheel_feedback_gain"] += 0.1
    else:
        value["source_hashes"]["rl/manipulation/control.py"] = "changed"
        evidence = component[-1]
        proof = json.loads(evidence.read_text())
        proof["source_hashes"] = value["source_hashes"]
        evidence.write_text(json.dumps(proof))
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        transfer(component)
