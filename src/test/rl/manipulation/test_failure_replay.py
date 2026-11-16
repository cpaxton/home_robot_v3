"""Failure neighborhoods preserve task balance, family splits and reset admission."""

from copy import deepcopy
from dataclasses import asdict, replace

import gymnasium as gym
import numpy as np
import pytest

from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.failure_replay import FailureReplay, family_partition, mine_failures, validate_corpus


def config():
    return ManipulationConfig(
        robot="innate_mars",
        scenario="mixed",
        horizon=12,
        randomization_scale=0,
        action_mode="direct_velocity",
        trajectory_preview=True,
        tracking_reward=True,
        retention_context=True,
        balanced_grasp_setup=True,
        fixed_base_contract=True,
        grasp_lift_probability=0.75,
        contact_cone="elliptic",
        contact_impratio=3,
        support_half_width=0.04,
        wheel_feedback_gain=0.5,
        physical_wheel_slew=True,
        whole_body_settling=True,
        hand_position_tolerance=0.003,
        hand_orientation_tolerance=0.02,
        payload_mass_kg=0.05,
    )


def development_report():
    cfg = replace(config(), scenario="carry", grasp_lift_probability=1, carry_push_probability=0)
    return {
        "complete": True,
        "controller": "policy",
        "config": asdict(cfg),
        "source_hashes": {},
        "cases": [
            {
                "seed": seed,
                "initialization_failed": False,
                "fingerprint": "fixture-model",
                "trace": [{}],
                "admission": {"accepted": True, "initial_state": {"qpos": [0], "qvel": [0], "ctrl": [0]}},
                "final_info": {"is_success": False, "code": "tracking_timeout"},
            }
            for seed in range(33_000_000, 33_000_020)
        ],
    }


def teacher_report():
    report = deepcopy(development_report())
    report["controller"] = "teacher"
    for case in report["cases"]:
        case["final_info"] = {"is_success": True, "code": "ok"}
    return report


def test_unverified_teacher_parents_are_preserved_but_excluded_from_debug_training():
    report, teacher = development_report(), teacher_report()
    teacher["cases"][0]["final_info"]["is_success"] = False
    corpus = mine_failures(report, teacher)
    assert corpus["teacher_unverified_parent_seeds"] == [33_000_000]
    assert 33_000_000 not in {f["seed"] for f in corpus["families"]}
    teacher["cases"][1]["admission"]["initial_state"]["qpos"] = [1]
    with pytest.raises(ValueError, match="physics differ"):
        mine_failures(report, teacher)


def test_corpus_splits_whole_parent_families_and_rejects_assessment_mining():
    report = development_report()
    corpus = mine_failures(report, teacher_report())
    assert {f["partition"] for f in corpus["families"]} == {"train", "validation"}
    validate_corpus(corpus, config(), check_sources=False)
    report["cases"][0]["seed"] = 3202000
    with pytest.raises(ValueError, match="assessment"):
        mine_failures(report, teacher_report())
    corpus["families"][0]["partition"] = (
        "validation" if family_partition(corpus["families"][0]["seed"]) == "train" else "train"
    )
    with pytest.raises(ValueError, match="families"):
        validate_corpus(corpus, config(), check_sources=False)


def test_corpus_preserves_rejections_and_does_not_mine_success_or_incomplete_reports():
    report = development_report()
    report["cases"][0]["initialization_failed"] = True
    report["cases"][1]["final_info"]["is_success"] = True
    corpus = mine_failures(report, teacher_report())
    assert corpus["rejected_parent_seeds"] == [33_000_000]
    assert corpus["successful_parent_seeds"] == [33_000_001]
    assert not {33_000_000, 33_000_001} & {f["seed"] for f in corpus["families"]}
    report["complete"] = False
    with pytest.raises(ValueError, match="complete"):
        mine_failures(report, teacher_report())


def test_replay_rejects_servo_or_load_drift():
    corpus = mine_failures(development_report(), teacher_report())
    for cfg in [replace(config(), physical_wheel_slew=False), replace(config(), payload_mass_kg=0.02)]:
        with pytest.raises(ValueError, match="configuration"):
            validate_corpus(corpus, cfg, check_sources=False)


class ResetRecorder(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (8,), dtype=np.float32)

    def __init__(self):
        self.config = config()
        self.fingerprint = "fixture-model"

    def reset(self, *, seed=None, options=None):
        self.recipe = (seed, options)
        return np.zeros(1, dtype=np.float32), {"scenario": self.config.scenario, "initialization_failed": False}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


def test_replay_preserves_default_outer_mix_and_never_draws_validation_parents(monkeypatch):
    from emet.rl.manipulation import carry_benchmark

    monkeypatch.setattr(carry_benchmark, "source_hashes", lambda: {})
    corpus = mine_failures(development_report(), teacher_report())
    env = FailureReplay(ResetRecorder(), corpus=corpus)
    counts = {"reach": 0, "carry": 0, "replay": 0}
    parents = set()
    first = None
    for i in range(20_000):
        _, info = env.reset(seed=104 if i == 0 else None)
        counts["reach"] += info["scenario"] == "reach"
        counts["carry"] += info["scenario"] == "carry"
        if info["failure_replay"]["selected"]:
            counts["replay"] += 1
            r = info["failure_replay"]
            parents.add(r["parent_seed"])
            assert family_partition(r["parent_seed"]) == "train"
            assert max(abs(np.asarray(r["goal_delta_rad"]))) <= 0.01
            assert env.unwrapped.config.grasp_lift_probability == 1
        if i == 0:
            first = (env.unwrapped.recipe, info)
    assert 0.24 < counts["reach"] / 20_000 < 0.26
    assert 0.29 < counts["carry"] / 20_000 < 0.31
    assert 0.14 < counts["replay"] / 20_000 < 0.16
    assert parents == {f["seed"] for f in corpus["families"] if f["partition"] == "train"}
    _, info = env.reset(seed=104)
    assert (env.unwrapped.recipe, info) == first
    assert "failure_replay" in env.step(np.zeros(8))[-1]
    env.close()


@pytest.mark.sim
def test_nearby_lift_target_replans_without_changing_reset_physics_or_grasp():
    env = ManipulationEnv(replace(config(), scenario="carry", grasp_lift_probability=1))
    try:
        _, a = env.reset(seed=3100002)
        assert not a["initialization_failed"]
        original_goal = np.array(env.grasp_lift_admission["goal"])
        initial = env.data.qpos.copy()
        velocity = env.data.qvel.copy()
        ctrl = env.data.ctrl.copy()
        retention = tuple(v.copy() for v in env.retention_reference)
        delta = [0, 0.001, 0, 0, 0]
        _, b = env.reset(seed=3100002, options={"lift_goal_delta": delta})
        assert not b["initialization_failed"], env.grasp_lift_admission
        np.testing.assert_array_equal(env.data.qpos, initial)
        np.testing.assert_array_equal(env.data.qvel, velocity)
        np.testing.assert_array_equal(env.data.ctrl, ctrl)
        np.testing.assert_allclose(env.grasp_lift_admission["goal"], original_goal + delta, atol=1e-12)
        for actual, expected in zip(env.retention_reference, retention, strict=True):
            np.testing.assert_array_equal(actual, expected)
        assert not env.control.release_authorized
        assert env.control.physical_wheel_slew_enabled
        assert env.grasp_lift_admission["minimum_sampled_margin_m"] >= 0.005
        before = env.data.qpos.copy()
        for bad in [[0] * 4, [float("nan")] * 5, [0.031] * 5]:
            with pytest.raises(ValueError, match="lift goal delta"):
                env.reset(options={"lift_goal_delta": bad})
            np.testing.assert_array_equal(env.data.qpos, before)
    finally:
        env.close()


@pytest.mark.sim
def test_perturbed_target_cannot_bypass_path_admission(monkeypatch):
    from emet.rl.manipulation import grasp_lift

    env = ManipulationEnv(replace(config(), scenario="carry", grasp_lift_probability=1))
    original = grasp_lift._executor

    def blocked(e):
        executor = original(e)
        monkeypatch.setattr(executor, "plan_joint_target", lambda goal: (None, "test_collision"))
        return executor

    monkeypatch.setattr(grasp_lift, "_executor", blocked)
    try:
        _, info = env.reset(seed=3100002, options={"lift_goal_delta": [0, 0.001, 0, 0, 0]})
        assert info["initialization_failed"] and env.control.stopped
        assert env.grasp_lift_admission["reason"] == "grasp_lift_plan:test_collision"
    finally:
        env.close()


@pytest.mark.parametrize("value", [1, "true", None])
def test_wheel_slew_requires_a_boolean(value):
    with pytest.raises(ValueError, match="wheel slew"):
        replace(config(), physical_wheel_slew=value)


def test_default_export_seed_is_in_the_mineable_development_namespace():
    import inspect

    from emet.rl.manipulation.failure_replay import DEVELOPMENT_END, DEVELOPMENT_START
    from emet.rl.manipulation.grasp_lift import DEFAULT_EXPORT_SEED, export_cases

    assert inspect.signature(export_cases).parameters["seed"].default == DEFAULT_EXPORT_SEED
    assert DEVELOPMENT_START <= DEFAULT_EXPORT_SEED < DEVELOPMENT_END
