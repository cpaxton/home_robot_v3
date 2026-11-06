"""Independent whole-body workers preserve physics and count every transition."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from emet.rl.manipulation.base_component import (
    BasePolicy,
    FrozenArmActions,
    MotionSamplingDiagnostics,
    initialize,
    make_motion_vec_env,
    merge,
    motion_rollout_size,
    train,
)
from emet.rl.manipulation.command_policy import ExecutorCommandVecNormalize, ExecutorFeedbackPolicy
from emet.rl.manipulation.feedback_policy import GoalFeedbackFeatures
from emet.rl.manipulation.mastery import MasteryEnv, load_suite
from emet.rl.manipulation.policy import ManipulationPolicy, export
from emet.rl.manipulation.whole_body import FrozenGripActions, WholeBodyPolicy


@pytest.mark.parametrize("count,backend", [(0, "dummy"), (9, "dummy"), (True, "dummy"), (2, "fork")])
def test_motion_worker_configuration_fails_before_model_or_simulator_load(count, backend):
    with pytest.raises(ValueError, match="1-8 motion environments"):
        make_motion_vec_env({}, [], "missing", num_envs=count, backend=backend)


@pytest.mark.parametrize("changes", [{"steps": 768}, {"evaluation_every": 768}, {"critic_warmup_steps": 256}])
def test_parallel_rollout_boundaries_fail_before_evidence_or_output_access(changes):
    args = SimpleNamespace(
        num_envs=2, vec_backend="subproc", steps=1024, evaluation_every=1024, critic_warmup_steps=512
    )
    vars(args).update(changes)
    with pytest.raises(ValueError, match=r"align to 256 \* num_envs"):
        train(args, {})
    assert motion_rollout_size(8, "subproc") == 2048


def test_motion_sampling_counts_workers_and_independent_auto_reset_starts():
    audit = MotionSamplingDiagnostics(2)
    info = {
        "mastery": {"case_id": "a", "base_velocity_command": [0, 0], "base_velocity_measured": [0, 0]},
        "initialization_failed": False,
        "base_goal_error": 0,
        "base_heading_error_rad": 0,
        "code": "ok",
    }
    other = deepcopy(info)
    other["mastery"]["case_id"] = "b"
    actions = np.zeros((2, 7))
    audit.add(actions, actions, [1, 2], [info, other], [False, True])
    audit.add(actions, actions, [3, 4], [info, other], [True, False])
    audit.add(actions, actions, [5, 6], [other, info], [False, False])
    assert dict(audit.steps) == {"a": 3, "b": 3}
    assert dict(audit.starts) == {"a": 1, "b": 3}
    rows = audit.rollout.record()
    assert sum(r["steps"] for r in rows.values()) == 6
    assert rows["a"]["reward_sum"] == 10 and rows["b"]["reward_sum"] == 11
    assert rows["a"]["terminal_codes"] == {"ok": 1}
    assert rows["b"]["terminal_codes"] == {"ok": 1}


@pytest.mark.sim
@pytest.mark.parametrize("train_arm", [False, True])
def test_two_native_motion_worlds_match_legacy_steps_update_and_export(tmp_path, train_arm):
    torch.set_num_threads(1)
    suite = load_suite(Path(__file__).resolve().parents[4] / "configs/benchmarks/mars_tracking_mastery_full_arm_v1.json")
    weights = {"arm": 1.0}
    native = MasteryEnv(suite, stages=["arm"], split="train", population="training", training_stage_weights=weights)
    config, fingerprint = native.config, native.fingerprint
    params = {
        "features_extractor_class": GoalFeedbackFeatures,
        "features_extractor_kwargs": {
            "robot": "innate_mars", "retention_context": True, "orientation_scale": config.feedback_orientation_scale
        },
        "restoring_yaw_feedback": True,
        "turn_xy_tolerance": config.base_xy_tolerance,
        "net_arch": {"pi": [16, 16], "vf": [16, 16]},
    }
    original = ExecutorCommandVecNormalize(DummyVecEnv([lambda: native]), config=config, norm_reward=False)
    vector = env = None
    try:
        source_model = PPO(ExecutorFeedbackPolicy, original, device="cpu", n_steps=8, batch_size=8, policy_kwargs=params)
        source_dir = tmp_path / "source"
        export(source_model, original, source_dir, config, fingerprint, steps=0)
        source = ManipulationPolicy(source_dir)
        wrapper = FrozenGripActions if train_arm else FrozenArmActions
        legacy = wrapper(native, source)
        vector = make_motion_vec_env(
            suite, ["arm"], source_dir, train_arm=train_arm, num_envs=2, backend="dummy", stage_weights=weights
        )
        # First worker reproduces the previous serial reset/step path exactly.
        vector.seed(90)
        observed = vector.reset()
        expected, _ = legacy.reset(seed=90)
        for k in expected:
            np.testing.assert_array_equal(observed[k][0], expected[k])
        workers = vector.get_attr("data")
        assert workers[0] is not workers[1]
        second_qpos = workers[1].qpos.copy()
        workers[0].qpos[0] += 0.001
        np.testing.assert_array_equal(workers[1].qpos, second_qpos)
        workers[0].qpos[0] -= 0.001
        action = np.zeros((2, 7 if train_arm else 2), dtype=np.float32)
        for _ in range(3):
            observed, reward, done, _ = vector.step(action)
            expected, old_reward, old_done, old_truncated, _ = legacy.step(action[0])
            assert reward[0] == np.float32(old_reward) and done[0] == (old_done or old_truncated)
            for k in expected:
                np.testing.assert_array_equal(observed[k][0], expected[k])
        env = ExecutorCommandVecNormalize(vector, config=config, norm_reward=False)
        model = PPO(
            WholeBodyPolicy if train_arm else BasePolicy,
            env,
            device="cpu", n_steps=8, batch_size=8, n_epochs=1, seed=90, use_sde=True, policy_kwargs=params,
        )
        initialize(model, env, source, config, fingerprint)
        before = model.policy.action_net.gain.bias.detach().clone()
        model.learn(16)
        assert model.num_timesteps == 16
        assert not torch.equal(before, model.policy.action_net.gain.bias)
        actor = merge(model, env, source, tmp_path / "actor", config, fingerprint)
        raw = env.get_original_obs()
        requested = model.predict(env.normalize_obs(raw), deterministic=True)[0]
        for i in range(2):
            obs = {k: v[i] for k, v in raw.items()}
            np.testing.assert_allclose(actor(obs)[:action.shape[1]], requested[i], atol=1e-7, rtol=0)
        assert actor.model.action_space.shape == (8,)
    finally:
        if env is not None:
            env.close()
        elif vector is not None:
            vector.close()
        original.close()
