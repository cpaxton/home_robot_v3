import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("stable_baselines3")
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from emet.rl.base import MarsBaseEnv
from emet.rl.env import EnvConfig
from emet.rl.policy import PPOPolicy
from emet.rl.training import TrainingMonitor, evaluation_metrics, finish_wandb, start_wandb

pytestmark = pytest.mark.sim


class Run:
    id = "test"
    url = "https://example.invalid/run"

    def __init__(self):
        self.rows = []
        self.finished = None
        self.definitions = []

    def log(self, row):
        self.rows.append(row)

    def define_metric(self, *args, **kwargs):
        self.definitions.append((args, kwargs))

    def finish(self, exit_code):
        self.finished = exit_code


def test_checkpoint_validation_and_best_policy(tmp_path):
    import torch

    torch.set_num_threads(1)
    env = VecNormalize(make_vec_env(MarsBaseEnv, n_envs=1, seed=0), norm_reward=False)
    run = Run()
    try:
        model = PPO("MultiInputPolicy", env, n_steps=16, batch_size=16, n_epochs=1, device="cpu")
        monitor = TrainingMonitor(
            tmp_path,
            EnvConfig(task="base", horizon_seconds=0.1),
            {"env": {"task": "base"}},
            interval=16,
            episodes=1,
            run=run,
        )
        model.learn(32, callback=monitor)
        assert (tmp_path / "step-16" / "validation.json").exists()
        assert (tmp_path / "step-32" / "validation.json").exists()
        policy = PPOPolicy(tmp_path / "best")
        assert policy.task == "base"
        assert any("validation/success_rate" in row for row in run.rows)
        steps = [row["timesteps"] for row in run.rows]
        assert steps == sorted(steps)
        assert steps.count(16) == 2  # Rollout and validation both retained at one simulator step.
    finally:
        env.close()


def test_wandb_optional_and_lifecycle(monkeypatch, tmp_path):
    import sys

    run = Run()
    calls = []
    fake = SimpleNamespace(init=lambda **kw: calls.append(kw) or run)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    args = SimpleNamespace(wandb_mode="disabled", wandb_project="test", wandb_entity=None, wandb_name="seed0")
    assert start_wandb(args, tmp_path, {}) is None
    assert not calls
    args.wandb_mode = "offline"
    assert start_wandb(args, tmp_path, {"seed": 0}) is run
    assert calls[0]["mode"] == "offline"
    assert "key" not in calls[0]
    assert json.loads((tmp_path / "wandb-run.json").read_text())["id"] == "test"
    assert run.definitions
    finish_wandb(run, tmp_path, failed=True)
    assert run.finished == 1


def test_no_success_has_no_fabricated_settling_time():
    report = {
        "success_rate": 0.0,
        "collision_rate": 0.0,
        "observation_and_inference_p95_ms": 1.0,
        "episodes": [
            {"is_success": False, "time": 10.0, "xy_error": 0.1, "yaw_error": 0.2, "rms_measured_jerk": [2.0, 3.0]}
        ],
    }
    metrics = evaluation_metrics(report)
    assert metrics["successful_episode_time"] is None
    assert metrics["rms_measured_jerk/angular"] == 3
    assert np.isfinite(metrics["xy_error"])


def test_bootstrap_exports_normalized_policy_and_bounded_labels(tmp_path):
    import torch

    from emet.rl.bootstrap import bootstrap_base

    torch.set_num_threads(1)
    env = VecNormalize(make_vec_env(MarsBaseEnv, n_envs=1, seed=3), norm_reward=False)
    args = SimpleNamespace(
        task="base",
        architecture="mlp",
        seeds=[3],
        bootstrap_rounds=2,
        bootstrap_steps=16,
        bootstrap_epochs=1,
        eval_episodes=1,
    )
    try:
        model = PPO("MultiInputPolicy", env, n_steps=16, batch_size=16, n_epochs=1, device="cpu")
        model = bootstrap_base(model, env, args, tmp_path, {"env": {"task": "base", "horizon_seconds": 0.1}})
        assert PPOPolicy(tmp_path / "bootstrap-best").task == "base"
        data = np.load(tmp_path / "demonstrations.npz")
        assert np.all(np.abs(data["actions"]) <= 1.0)
        assert len(data["actions"]) == 32
        np.testing.assert_allclose(model.policy.log_std.detach().numpy(), -3)
    finally:
        env.close()


def test_base_features_use_current_goal_and_past_motion():
    import torch

    from emet.rl.networks import BaseHistoryFeatures

    env = MarsBaseEnv()
    try:
        obs, _ = env.reset(seed=0)
        extractor = BaseHistoryFeatures(env.observation_space)
        history = torch.as_tensor(obs["history"][None])
        before = extractor({"history": history}).clone()
        history[:, :-1, :4] += 10  # Old targets must not delay a new goal.
        torch.testing.assert_close(extractor({"history": history}), before)
        history[:, -1, 0] += 1
        assert not torch.equal(extractor({"history": history}), before)
        before = extractor({"history": history}).clone()
        history[:, 0, 4] += 1  # Past measured motion must remain available.
        assert not torch.equal(extractor({"history": history}), before)
    finally:
        env.close()


def test_precision_policy_resume_keeps_normalization_frozen(tmp_path):
    import torch

    from emet.rl.__main__ import train
    from emet.rl.networks import PrecisionBaseFeatures
    from emet.rl.policy import export_policy

    torch.set_num_threads(1)
    env = VecNormalize(make_vec_env(MarsBaseEnv, n_envs=1, seed=0), norm_reward=False)
    try:
        env.reset()
        env.training = False
        model = PPO(
            "MultiInputPolicy",
            env,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
            device="cpu",
            policy_kwargs={
                "features_extractor_class": PrecisionBaseFeatures,
                "features_extractor_kwargs": {"normalized_goal_threshold": 0.0},
            },
        )
        export_policy(model, env, tmp_path / "initial", {"env": {"task": "base"}})
        initial = json.loads((tmp_path / "initial" / "manifest.json").read_text())
    finally:
        env.close()
    args = SimpleNamespace(
        task="base",
        architecture="mlp",
        stage=0,
        contact_training=False,
        seeds=[0],
        output=str(tmp_path / "continued"),
        envs=1,
        resume=str(tmp_path / "initial"),
        steps=16,
        learning_rate=None,
        clip_range=None,
        critic_steps=0,
        bootstrap_steps=0,
        eval_every=16,
        eval_episodes=1,
        wandb_mode="disabled",
    )
    train(args)
    final = json.loads((tmp_path / "continued" / "seed-0" / "manifest.json").read_text())
    assert final["normalization_frozen"]
    assert initial["normalization"] == final["normalization"]


def test_critic_warmup_changes_values_not_actor():
    import torch

    from emet.rl.critic import warmup_critic

    torch.set_num_threads(1)
    env = VecNormalize(
        make_vec_env(MarsBaseEnv, n_envs=1, seed=0, env_kwargs={"config": EnvConfig(task="base", horizon_seconds=0.1)}),
        norm_reward=False,
    )
    try:
        model = PPO("MultiInputPolicy", env, n_steps=16, batch_size=16, n_epochs=1, device="cpu")
        before_actor = {k: v.clone() for k, v in model.policy.action_net.state_dict().items()}
        before_latent = {k: v.clone() for k, v in model.policy.mlp_extractor.policy_net.state_dict().items()}
        before_value = model.policy.value_net.weight.detach().clone()
        metrics = warmup_critic(model, env, 32, epochs=2)
        for k, v in before_actor.items():
            torch.testing.assert_close(v, model.policy.action_net.state_dict()[k])
        for k, v in before_latent.items():
            torch.testing.assert_close(v, model.policy.mlp_extractor.policy_net.state_dict()[k])
        assert not torch.equal(before_value, model.policy.value_net.weight)
        assert metrics["critic_warmup/examples"] == 32
    finally:
        env.close()
