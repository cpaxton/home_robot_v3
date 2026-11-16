"""The arm diagnostic changes action authority, not goals or physical guards."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from emet.rl.manipulation.arm_component import SCOPE, ArmOnlyActions, ArmPolicy, expand_action, require_preflight
from emet.rl.manipulation.carry_benchmark import digest, source_hashes
from emet.rl.manipulation.feedback_policy import GoalFeedbackFeatures
from emet.rl.manipulation.mastery import MasteryEnv, load_suite, select_cases
from emet.rl.manipulation.normalization import CommandVecNormalize


def suite():
    return load_suite(
        Path(__file__).resolve().parents[4] / "configs/benchmarks/mars_tracking_mastery_hand_accuracy_v1.json"
    )


@pytest.mark.parametrize("bad", [[0] * 8, [0] * 4, [float("nan")] * 5, [1.1] * 5])
def test_arm_component_rejects_invalid_rates(bad):
    with pytest.raises(ValueError):
        expand_action(bad)


def test_arm_component_preserves_arm_commands_and_masks_base_and_grip():
    action = np.linspace(-1, 1, 5)
    full = expand_action(action)
    np.testing.assert_array_equal(full[2:-1], action)
    np.testing.assert_array_equal(full[[0, 1, 7]], 0)


def test_arm_component_teacher_gate_rejects_wrong_scope_sources_and_missing_cases():
    spec = suite()
    report = {
        "format": SCOPE,
        "controller": "teacher",
        "complete": True,
        "source_hashes": source_hashes(),
        "suite_sha256": digest(spec),
        "cases": [
            {"id": c["id"], "mastered": True} for c in select_cases(spec, ["arm"], "all", population="admission")
        ],
    }
    require_preflight(report, spec)
    for key, value in [
        ("format", "whole-body"),
        ("source_hashes", {}),
        ("suite_sha256", "different"),
        ("cases", report["cases"][:-1]),
        ("cases", [{**c, "mastered": False} for c in report["cases"]]),
    ]:
        changed = deepcopy(report)
        changed[key] = value
        with pytest.raises(ValueError):
            require_preflight(changed, spec)


@pytest.mark.sim
def test_arm_component_actual_ppo_updates_roundtrip_and_keeps_other_requests_fixed(tmp_path):
    torch.set_num_threads(1)
    base = MasteryEnv(suite(), stages=["arm"], split="train", population="training")
    env = CommandVecNormalize(DummyVecEnv([lambda: ArmOnlyActions(base)]), config=base.config, norm_reward=False)
    try:
        model = PPO(
            ArmPolicy,
            env,
            device="cpu",
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            seed=19,
            policy_kwargs={
                "features_extractor_class": GoalFeedbackFeatures,
                "features_extractor_kwargs": {"robot": "innate_mars", "retention_context": True},
            },
        )
        obs = env.reset()
        before = model.policy.action_net.gain.weight.detach().clone()
        model.learn(16)
        assert model.num_timesteps == 16 and not torch.equal(before, model.policy.action_net.gain.weight)
        assert base.control.base_mask.all() and base.control.hand_mask
        assert base.action_space.shape == (8,) and env.action_space.shape == (5,)
        np.testing.assert_array_equal(base.control.base_command, 0)
        assert base.control.grip_target == pytest.approx(base.control.grip_reference)
        expected, _ = model.predict(obs, deterministic=True)
        path = tmp_path / "actor.zip"
        model.save(path)
        restored = PPO.load(path, device="cpu")
        np.testing.assert_allclose(restored.predict(obs, deterministic=True)[0], expected, atol=1e-7)
        assert not (tmp_path / "manifest.json").exists()
    finally:
        env.close()
