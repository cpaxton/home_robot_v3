"""Restoring means, opposing damping, mode continuity and real PPO transfer."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from emet.rl.manipulation.env import ManipulationConfig
from emet.rl.manipulation.feedback_policy import (
    FeedbackAction,
    FeedbackActorCriticPolicy,
    GoalFeedbackFeatures,
    RestoringYawAction,
)
from emet.rl.manipulation.initialization import transfer_actor
from emet.rl.manipulation.mastery import MasteryEnv, load_suite, select_cases
from emet.rl.manipulation.normalization import CommandVecNormalize
from emet.rl.manipulation.policy import ManipulationPolicy, export


@pytest.mark.parametrize("direction", [-1, 1])
def test_restoring_mean_at_rest_and_independent_damping_allow_braking(direction):
    torch.manual_seed(10)
    head = RestoringYawAction(128, 35, 8)
    with torch.no_grad():
        head.gain.weight.normal_(std=0.3)
        head.gain.bias.normal_(std=0.3)
        head.turn_gain.weight.normal_(std=0.05)
    context = torch.randn(32, 128)
    signal = torch.zeros(32, 35)
    signal[:, [2, 14]] = direction * 0.08
    latent = torch.cat([context, torch.ones(32, 1), signal], dim=1)
    action = head(latent)
    assert (direction * action[:, 1] > 0).all()
    signal.zero_()
    signal[:, -1] = direction * 0.1
    damped = head(torch.cat([context, torch.ones(32, 1), signal], dim=1))
    assert (direction * damped[:, 1] < 0).all()
    # Positive position error can legitimately require negative command to brake.
    head.turn_gain.weight.data.zero_()
    signal.zero_()
    signal[:, [2, 14]] = 0.08
    signal[:, -1] = 0.5
    assert (head(torch.cat([context, torch.ones(32, 1), signal], dim=1))[:, 1] < 0).all()
    signal.zero_()
    assert torch.equal(head(torch.cat([context, torch.ones(32, 1), signal], dim=1))[:, :-1], torch.zeros(32, 7))


def test_general_mode_and_non_yaw_outputs_are_preserved_exactly():
    torch.manual_seed(11)
    old = FeedbackAction(128, 35, 8)
    new = RestoringYawAction(128, 35, 8)
    with torch.no_grad():
        for parameter in old.parameters():
            parameter.normal_(std=0.2)
        new.gain.load_state_dict(old.gain.state_dict())
        new.grip.load_state_dict(old.grip.state_dict())
    context = torch.randn(8, 128)
    signals = torch.randn(8, 35)
    expected = old(torch.cat([context, signals], dim=1))
    off = new(torch.cat([context, torch.zeros(8, 1), signals], dim=1))
    assert torch.equal(off, expected)
    on = new(torch.cat([context, torch.ones(8, 1), signals], dim=1))
    assert torch.equal(on[:, [0, 2, 3, 4, 5, 6, 7]], expected[:, [0, 2, 3, 4, 5, 6, 7]])


def test_smooth_gate_uses_all_preview_targets_and_masks():
    import gymnasium as gym

    space = gym.spaces.Dict(
        {
            "history": gym.spaces.Box(-np.inf, np.inf, (20, 101), dtype=np.float32),
            "preview": gym.spaces.Box(-np.inf, np.inf, (7, 18), dtype=np.float32),
        }
    )
    features = GoalFeedbackFeatures(space, robot="innate_mars", retention_context=True)
    obs = {"history": torch.zeros(5, 20, 101), "preview": torch.zeros(5, 7, 18)}
    obs["preview"][:, :, 6] = 1
    obs["preview"][:, :, 1] = torch.tensor([0, 0.005, 0.0075, 0.01, 0.02])[:, None] / 0.1
    gate = features.turn_gate(features(obs), 0.005)[:, 0]
    torch.testing.assert_close(gate, torch.tensor([1, 1, 0.5, 0, 0]), atol=1e-6, rtol=0)
    obs["preview"][0, -1, 1] = 0.2
    obs["preview"][1, :, 16] = 1
    obs["preview"][2, :, 6] = 0
    assert not features.turn_gate(features(obs), 0.005)[:3].any()


def test_restoring_suite_preserves_cases_gates_reward_and_actuation():
    root = Path(__file__).resolve().parents[4] / "configs/benchmarks"
    old = load_suite(root / "mars_tracking_mastery_yaw_curriculum_v1.json")
    new = load_suite(root / "mars_tracking_mastery_restoring_yaw_v1.json")
    assert new["config"].pop("restoring_yaw_feedback") is True
    assert new["config"] == old["config"]
    for key in ["cases", "training_population", "reference", "gate"]:
        assert new[key] == old[key]
    assert len(select_cases(new, ["base"], "train", population="training")) == 45
    with pytest.raises(ValueError):
        ManipulationConfig(restoring_yaw_feedback=True)


@pytest.mark.sim
def test_real_ppo_transfer_keeps_shared_weights_then_trains_and_roundtrips(tmp_path):
    from emet.rl.manipulation.__main__ import resume_training
    from emet.rl.manipulation.yaw_audit import feedback_terms

    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[4]
    suite = load_suite(root / "configs/benchmarks/mars_tracking_mastery_restoring_yaw_v1.json")
    cfg = ManipulationConfig(**suite["config"])
    oldcfg = replace(cfg, restoring_yaw_feedback=False)
    env = CommandVecNormalize(DummyVecEnv([lambda: MasteryEnv(suite, stages=["base"])]), config=cfg, norm_reward=False)

    def model(restoring):
        return PPO(
            FeedbackActorCriticPolicy,
            env,
            device="cpu",
            n_steps=16,
            batch_size=8,
            n_epochs=1,
            seed=9,
            policy_kwargs={
                "restoring_yaw_feedback": restoring,
                "turn_xy_tolerance": cfg.base_xy_tolerance,
                "features_extractor_class": GoalFeedbackFeatures,
                "features_extractor_kwargs": {"robot": cfg.robot, "retention_context": True, "orientation_scale": 0.15},
                "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            },
        )

    try:
        old = model(False)
        fp = env.get_attr("fingerprint")[0]
        export(old, env, tmp_path / "old", oldcfg, fp, steps=0)
        new = model(True)
        details = transfer_actor(new, env, tmp_path / "old", cfg, fp)
        assert details["restoring_yaw_feedback_change"] == {"from": False, "to": True, "function_preserving": False}
        for key, value in old.policy.state_dict().items():
            assert torch.equal(value, new.policy.state_dict()[key]), key
        assert not new.policy.optimizer.state_dict()["state"]
        new.learn(32)
        assert new.policy.optimizer.state_dict()["state"]
        export(new, env, tmp_path / "new", cfg, fp, steps=32)
        loaded = ManipulationPolicy(tmp_path / "new")
        assert loaded.model.policy.restoring_yaw_feedback
        raw, _ = env.venv.envs[0].reset(options={"case_id": "train:base:yaw_left"})
        action, row = feedback_terms(loaded, raw)
        np.testing.assert_array_equal(action, loaded(raw))
        assert row["restoring_gate"] == 1 and min(row["restoring_gains"]) > 0
        with pytest.raises(ValueError, match="semantics mismatch"):
            resume_training(tmp_path / "new", env, oldcfg, fp, "mlp")
        assert resume_training(tmp_path / "new", env, cfg, fp, "mlp").policy.restoring_yaw_feedback
    finally:
        env.close()
