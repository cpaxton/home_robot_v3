"""Training-only lift of a verified five-output arm actor into one whole-body actor."""

import hashlib
import json
import pickle
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from emet.rl.manipulation.arm_component import SCOPE
from emet.rl.manipulation.carry_benchmark import digest, source_hashes
from emet.rl.manipulation.feedback_policy import FeedbackAction, GoalFeedbackFeatures
from emet.rl.manipulation.initialization import _model
from emet.rl.manipulation.mastery import read_report
from emet.rl.manipulation.normalization import CommandVecNormalize


def _checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transfer_arm_component(model, normalizer, directory, evidence, config, fingerprint):
    """Preserve arm/critic means and normalization; initialize new base/grip rows.

    The producer's final checkpoint remains immutable. Checksummed evidence is
    checked before deserializing our own PPO/statistics artifacts. This is not
    runtime admission: current teacher/regression gates are still mandatory.
    """
    directory, evidence = Path(directory), Path(evidence)
    if directory.name != "last":
        raise ValueError("arm transfer requires the final component checkpoint")
    proof = json.loads(evidence.read_text())
    metadata = json.loads((directory.parent / "component.json").read_text())
    complete = json.loads((directory.parent / "complete.json").read_text())
    if (
        proof.get("scope") != SCOPE
        or metadata.get("scope") != SCOPE
        or metadata.get("action_size") != 5
        or metadata.get("production_admitted") is not False
        or proof.get("production_admitted") is not False
        or proof.get("steps") != metadata.get("steps")
        or proof.get("steps") != complete.get("steps")
        or not isinstance(proof.get("steps"), int)
        or proof["steps"] <= 0
        or proof.get("source_hashes") != metadata.get("source_hashes")
    ):
        raise ValueError("not complete, consistently bound arm-component evidence")
    checked = {}
    for name in ("actor.zip", "normalizer.pkl", "evaluation.json.gz"):
        entries = [
            value
            for key, value in proof.get("artifact_checksums", {}).items()
            if Path(key).name == name and Path(key).parent.name == "last"
        ]
        if len(entries) != 1 or entries[0] != _checksum(directory / name):
            raise ValueError("component artifact checksum mismatch: " + name)
        checked[name] = entries[0]
    report = read_report(directory / "evaluation.json.gz")
    if (
        report.get("format") != SCOPE
        or report.get("controller") != "ppo"
        or not report.get("complete")
        or report.get("source_hashes") != metadata["source_hashes"]
        or report.get("suite_sha256") != digest(metadata["suite"])
        or len(report.get("cases", [])) != 8
        or {r["id"] for r in report["cases"]} != {c["id"] for c in metadata["suite"]["cases"] if c["stage"] == "arm"}
        or not all(r.get("admitted") and r.get("mastered") for r in report["cases"])
    ):
        raise ValueError("verified final canonical component successes required")
    if json.loads(json.dumps(asdict(config))) != metadata["env"] or config.robot != "innate_mars":
        raise ValueError("component transfer requires identical environment and control configuration")
    current = source_hashes()
    # Training/evaluation helpers may evolve; geometry, controls, observation
    # encoding, normalization and deployed mean parameterization may not drift.
    exact = (
        "assets/robot/",
        "robots/",
        "rl/control.py",
        "rl/contact.py",
        "rl/model.py",
        "rl/profile.py",
        "rl/reference.py",
        "rl/manipulation/models.py",
        "rl/manipulation/arm_component.py",
        "rl/manipulation/control.py",
        "rl/manipulation/profiles.py",
        "rl/manipulation/env.py",
        "rl/manipulation/observation_layout.py",
        "rl/manipulation/tracking.py",
        "rl/manipulation/normalization.py",
        "rl/manipulation/feedback_policy.py",
        "rl/manipulation/policy.py",
    )
    if any(metadata["source_hashes"].get(k) != v for k, v in current.items() if k.startswith(exact)):
        raise ValueError("component geometry/control/observation sources changed")
    _, reconstructed = _model(config.robot, asdict(config))
    if reconstructed != fingerprint:
        raise ValueError("component target model mismatch")
    source = PPO.load(directory / "actor.zip", device="cpu")
    # The statistics pickle is locally produced and verified above, never an
    # untrusted remote candidate. Do not attach its five-action VecEnv to target.
    with (directory / "normalizer.pkl").open("rb") as stream:
        old_normalizer = pickle.load(stream)
    if (
        not isinstance(old_normalizer, CommandVecNormalize)
        or source.action_space.shape != (5,)
        or model.action_space.shape != (8,)
        or not isinstance(source.policy.features_extractor, GoalFeedbackFeatures)
        or not isinstance(model.policy.features_extractor, GoalFeedbackFeatures)
        or not isinstance(model.policy.action_net, FeedbackAction)
        or source.policy.features_extractor.orientation_scale != model.policy.features_extractor.orientation_scale
        or source.policy.features_extractor.feedback_width != model.policy.features_extractor.feedback_width
    ):
        raise ValueError("incompatible component actor/feature contract")
    if set(old_normalizer.obs_rms) != set(normalizer.obs_rms):
        raise ValueError("component normalization keys differ")
    for key, stats in old_normalizer.obs_rms.items():
        if stats.mean.shape != normalizer.obs_rms[key].mean.shape:
            raise ValueError("component observation layout differs")
    zero_columns = list(getattr(normalizer, "component_zero_columns", []))
    if zero_columns and np.any(old_normalizer.obs_rms["history"].mean[..., zero_columns] != 0):
        raise ValueError("command-unit migration requires zero command means in the arm component")
    old, new = source.policy.state_dict(), model.policy.state_dict()
    shared = {key for key in new if not key.startswith("action_net.") and key != "log_std"}
    if shared != {key for key in old if not key.startswith("action_net.") and key != "log_std"}:
        raise ValueError("component actor/critic trunk differs")
    for key in shared:
        if old[key].shape != new[key].shape:
            raise ValueError("component trunk parameter layout differs: " + key)
        new[key] = old[key].clone()
    width = source.policy.features_extractor.feedback_width
    for key in ("action_net.gain.weight", "action_net.gain.bias"):
        target = torch.zeros_like(new[key]).reshape(7, width, -1)
        target[2:] = old[key].reshape(5, width, -1)
        new[key] = target.reshape(new[key].shape)
    for key in ("action_net.grip.weight", "action_net.grip.bias"):
        new[key].zero_()
    model.policy.load_state_dict(new)
    normalizer.obs_rms = deepcopy(old_normalizer.obs_rms)
    normalizer.epsilon, normalizer.clip_obs = old_normalizer.epsilon, old_normalizer.clip_obs
    normalizer.pin_command_features()
    normalization_exact = all(
        np.array_equal(stats.mean, normalizer.obs_rms[key].mean)
        and np.array_equal(stats.var, normalizer.obs_rms[key].var)
        for key, stats in old_normalizer.obs_rms.items()
    )
    if any(_checksum(directory / name) != value for name, value in checked.items()):
        raise RuntimeError("component artifacts changed during transfer")
    return {
        "kind": "arm_component_to_whole_body",
        "source_artifacts": checked,
        "evidence_sha256": _checksum(evidence),
        "source_fingerprint": reconstructed,
        "arm_mean_function_preserving": True,
        "critic_function_preserving": True,
        "new_base_general_mean": "zero",
        "new_yaw_branch": "target constructor initialization",
        "grip_mean": "zero offset",
        "normalization_copied_exactly": normalization_exact,
        "zero_input_command_unit_columns": zero_columns if not normalization_exact else [],
        "function_preserving_domain": "zero-wheel, zero-grip component observations"
        if not normalization_exact
        else "all observations",
        "optimizer_restored": False,
        "runtime_admitted": False,
        "safety_promoted": False,
    }
