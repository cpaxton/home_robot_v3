"""Physical, repeatable probes of goal changes, disturbances and contact recovery."""

import numpy as np
import pytest

pytest.importorskip("mujoco")
from emet.rl.base import MarsBaseEnv
from emet.rl.env import EnvConfig

pytestmark = pytest.mark.sim


@pytest.mark.parametrize("scenario", ["retarget", "push", "contact_retreat"])
def test_teacher_recovers_after_scheduled_event(scenario):
    env = MarsBaseEnv(EnvConfig(task="base", base_scenario=scenario, contact_training=True, horizon_seconds=15))
    try:
        env.reset(seed=71000)
        contact, applied_push, peak = False, False, 0.0
        goal_before = env.control.goal.base.copy()
        for _ in range(300):
            _, _, done, truncated, info = env.step(env.expert_action())
            applied_push |= bool(np.any(env.data.xfrc_applied))
            contact |= info["collision"]
            peak = max(peak, info["peak_contact_force"])
            if info["event_count"] == 0:
                assert not info["is_success"]
            if done or truncated:
                break
        assert info["event_count"] == 1
        assert info["is_success"], info
        assert not info["unsafe_contact"]
        assert info["event_elapsed"] > 0
        if scenario == "push":
            assert applied_push
            assert not env.data.xfrc_applied.any()
        else:
            assert not np.allclose(goal_before, env.control.goal.base)
        if scenario == "contact_retreat":
            assert contact and peak > 1
            assert info["peak_contact_force"] < 1
        first = env.reset(seed=71000)[0]["history"].copy()
        second = env.reset(seed=71000)[0]["history"]
        np.testing.assert_array_equal(first, second)
    finally:
        env.close()


def test_hard_impact_stops_before_scheduled_retreat():
    env = MarsBaseEnv(
        EnvConfig(
            task="base",
            base_scenario="contact_retreat",
            contact_training=True,
            unsafe_contact_force=1,
            horizon_seconds=15,
        )
    )
    try:
        env.reset(seed=71000)
        for _ in range(300):
            _, _, done, truncated, info = env.step(env.expert_action())
            if done or truncated:
                break
        assert done and info["unsafe_contact"] and not info["is_success"]
        assert info["event_count"] == 0
        np.testing.assert_array_equal(env.control.last_action, 0)
        np.testing.assert_array_equal(env.data.ctrl[env.control.base_act], 0)
    finally:
        env.close()


def test_contact_benchmark_exposes_policy_to_contact_before_handoff():
    from emet.rl.__main__ import evaluate

    class WaitPolicy:
        task = "base"

        def __call__(self, obs):
            return np.zeros(2)

    report = evaluate(
        WaitPolicy(),
        EnvConfig(task="base", base_scenario="contact_retreat", contact_training=True, horizon_seconds=15),
        episodes=1,
        seed=71000,
        teacher_approach=True,
    )
    row = report["episodes"][0]
    assert report["teacher_approach"]
    assert row["peak_contact_force"] > 1 and row["collision"]
    assert row["event_count"] == 1
    assert not row["is_success"]  # Waiting at the box cannot count as recovery.


def test_scenario_reset_clears_old_obstacle_before_sampling(monkeypatch):
    env = MarsBaseEnv(EnvConfig(task="base", base_scenario="mixed"))
    original = env._sample_configuration

    def sample():
        assert np.all(env.data.mocap_pos[:, :2] == 5)
        return original()

    monkeypatch.setattr(env, "_sample_configuration", sample)
    try:
        for seed in range(30):
            env.reset(seed=seed)
    finally:
        env.close()


def test_residual_feedback_prior_matches_teacher_across_goals():
    torch = pytest.importorskip("torch")
    from emet.rl.residual import ResidualBaseFeatures

    env = MarsBaseEnv()
    try:
        env.reset(seed=42)
        width = env.observation_space["history"].shape[-1]
        features = ResidualBaseFeatures(env.observation_space, np.zeros(width), np.ones(width))
        start = env.control.base_pose()
        for distance in [0, 0.005, 0.0119, 0.0121, 0.2, 0.6]:
            for angle in [-2.9, -1.0, 0.0, 1.0, 2.9]:
                env.set_goal({"base": [start[0] + distance * np.cos(angle), start[1] + distance * np.sin(angle), -0.4]})
                obs = env.control.observe()[0]
                prior = features({"history": torch.as_tensor(obs["history"])[None]})[0, -2:].numpy()
                np.testing.assert_allclose(prior, env.expert_action(), atol=1e-5)
    finally:
        env.close()


def test_mixed_ordinary_deadline_matches_ten_second_gate():
    for scenario, expected in (("mixed", True), ("standard", False)):
        env = MarsBaseEnv(EnvConfig(task="base", base_scenario=scenario, horizon_seconds=15))
        try:
            env.reset(seed=19)
            env._scenario = "standard"
            env.set_goal({"base": (env.control.base_pose() + [1, 0, 0]).tolist()})
            env.steps = 199
            _, _, terminated, truncated, _ = env.step(np.zeros(2))
            assert not terminated
            assert truncated is expected
        finally:
            env.close()


def test_contact_enabled_arrival_does_not_require_zero_contact(monkeypatch):
    env = MarsBaseEnv(EnvConfig(task="base", contact_training=True))
    try:
        env.reset(seed=42)
        observe = env.control.observe

        def touching():
            obs, metrics = observe()
            return obs, {**metrics, "collision": True}

        monkeypatch.setattr(env.control, "observe", touching)
        monkeypatch.setattr(env.control, "settled", lambda metrics, dt: True)
        _, _, terminated, _, info = env.step(np.zeros(2))
        assert terminated and info["is_success"]
        assert info["collision"] and not info["unsafe_contact"]
    finally:
        env.close()


def test_checkpoint_selection_uses_contact_severity_not_contact_count():
    from emet.rl.training import checkpoint_score

    gentle = {
        "success_rate": 1,
        "collision_rate": 1,
        "unsafe_contact_rate": 0,
        "xy_error": 0.01,
        "yaw_error": 0,
        "peak_contact_force": 2,
        "contact_impulse": 0.1,
    }
    no_contact_label = {**gentle, "collision_rate": 0}
    harsh = {**no_contact_label, "unsafe_contact_rate": 0.1, "peak_contact_force": 40}
    assert checkpoint_score(gentle, contact_training=True) == checkpoint_score(no_contact_label, contact_training=True)
    assert checkpoint_score(gentle, contact_training=True) > checkpoint_score(harsh, contact_training=True)
    jerky = {**gentle, "rms_measured_jerk/linear": 20}
    assert checkpoint_score(gentle, contact_training=True) > checkpoint_score(jerky, contact_training=True)
