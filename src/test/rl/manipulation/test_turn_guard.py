"""Signed yaw and divergence protections must preserve braking and masks."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from emet.rl.manipulation.env import ManipulationConfig
from emet.rl.manipulation.mastery import MasteryEnv, load_suite
from emet.rl.manipulation.policy import ManipulationPolicy
from emet.rl.manipulation.tracking import encode_preview
from emet.rl.manipulation.turn_guard import TurnProgressGuard
from emet.rl.manipulation.yaw_audit import sample_suite


@pytest.mark.parametrize("direction", [-1, 1])
def test_wrapped_target_minus_measured_yaw_keeps_sign_through_normalization(direction):
    measured = direction * (np.pi - 0.02)
    target = -measured
    identity = np.r_[np.zeros(3), [1, 0, 0, 1, 0, 0]]
    control = SimpleNamespace(
        base_pose=lambda: np.array([0.0, 0.0, measured]),
        ee_pose=lambda: identity,
        base_goal=np.array([0.0, 0.0, target]),
        hand_goal=identity,
        base_mask=np.ones(3),
        hand_mask=False,
        data=SimpleNamespace(time=0.0, qpos=np.zeros(1)),
        grip_q=0,
        release_authorized=False,
        grip_reference=0.0,
    )
    obs = {"preview": encode_preview(control)}
    np.testing.assert_allclose(obs["preview"][:, 3], direction * 0.04, atol=1e-7)
    fake = SimpleNamespace(
        normalization={"preview": (np.zeros((7, 18)), np.full((7, 18), 0.25 - 1e-8))},
        manifest={"epsilon": 1e-8, "clip": 20},
    )
    normalized = ManipulationPolicy.normalized_observation(fake, obs)
    np.testing.assert_allclose(normalized["preview"][:, 3], direction * 0.08, atol=1e-7)
    control.base_mask[2] = 0
    assert not encode_preview(control)[:, 3].any()


@pytest.mark.parametrize("sign", [-1, 1])
def test_wrong_way_command_and_measured_motion_latch_after_persistence(sign):
    guard = TurnProgressGuard()
    for i in range(21):
        fault = guard.update(
            time=i * 0.05,
            yaw=-sign * 0.01 * i * 0.05,
            target=sign * 0.04,
            command=-sign * 0.01,
            eligible=True,
            tolerance=0.005,
        )
        if fault:
            break
    assert fault == "base_turn_divergence"
    assert guard.evidence["duration_s"] >= 0.5 and guard.evidence["away_rad"] >= 0.0025
    assert (
        guard.update(time=2, yaw=0.04 * sign, target=0.04 * sign, command=0, eligible=False, tolerance=0.005) == fault
    )
    guard.reset()
    assert guard.fault is None


@pytest.mark.parametrize("mode", ["braking", "disturbance", "moving_target", "ineligible", "small_error"])
def test_monitor_does_not_mistake_valid_or_inapplicable_motion_for_divergence(mode):
    guard = TurnProgressGuard()
    for i in range(60):
        t = i * 0.05
        yaw = -0.01 * t
        command = -0.01
        target = 0.04
        eligible = True
        if mode == "braking":
            yaw = 0.01 * t
        elif mode == "disturbance":
            command = 0
        elif mode == "moving_target":
            target = 0.04 + 0.01 * t
        elif mode == "ineligible":
            eligible = False
        elif mode == "small_error":
            target = yaw + 0.005
        assert guard.update(time=t, yaw=yaw, target=target, command=command, eligible=eligible, tolerance=0.005) is None


def test_repeated_physics_time_is_not_a_new_motion_sample_and_invalid_input_fails_closed():
    guard = TurnProgressGuard()
    values = {"time": 0.0, "yaw": 0.0, "target": 0.04, "command": -0.01, "eligible": True, "tolerance": 0.005}
    assert guard.update(**values) is None
    assert guard.update(**values) is None
    assert guard.update(**{**values, "yaw": np.nan}) == "base_turn_invalid_feedback"
    with pytest.raises(ValueError):
        ManipulationConfig(base_turn_progress_guard=1)


def test_yaw_sampling_uses_only_train_postures_and_covers_signs_and_scales():
    parent = load_suite()
    sample = sample_suite(parent)
    assert len(sample["cases"]) == 30
    assert {c["turn_rad"] for c in sample["cases"]} == {-0.06, -0.04, -0.02, 0.02, 0.04, 0.06}
    seeds = {c["reset_seed"] for c in sample["cases"]}
    assert seeds == {c["reset_seed"] for c in parent["cases"] if c["stage"] == "base" and c["split"] == "train"}
    assert seeds.isdisjoint(c["reset_seed"] for c in parent["cases"] if c["split"] == "validation")
    assert parent["config"].get("base_turn_progress_guard") is None
    for values in [[0], [np.nan], [0.1], [0.04, 0.04]]:
        with pytest.raises(ValueError):
            sample_suite(parent, values)


@pytest.mark.sim
def test_real_wrong_way_rollout_stops_and_cannot_rearm_or_release_grip():
    suite = deepcopy(load_suite())
    suite["config"]["base_turn_progress_guard"] = True
    env = MasteryEnv(suite, stages=["base"], split="train")
    try:
        _, info = env.reset(options={"case_id": "train:base:yaw_left"})
        assert not info["initialization_failed"]
        grip = env.control.grip_reference
        action = np.zeros(8)
        action[1] = -0.01
        for _ in range(240):
            _, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        assert done and not truncated and info["code"] == "base_turn_divergence"
        assert not info["is_success"] and env.control.stopped
        assert info["turn_guard_evidence"]["away_rad"] >= 0.0025
        np.testing.assert_array_equal(env.control.base_command, [0, 0])
        assert env.control.grip_reference == grip and not env.control.release_authorized
        with pytest.raises(ValueError, match="latched motion fault"):
            env.control.rearm()
        env.control.turn_progress_guard_enabled = False
        with pytest.raises(ValueError):
            env.control.rearm()
    finally:
        env.close()


def test_training_yaw_replay_preserves_originals_and_requires_full_teacher_evidence():
    from pathlib import Path

    from emet.rl.manipulation.carry_benchmark import digest, source_hashes
    from emet.rl.manipulation.mastery import select_cases, validate_preflight

    root = Path(__file__).resolve().parents[4]
    suite = load_suite(root / "configs/benchmarks/mars_tracking_mastery_yaw_curriculum_v1.json")
    crossed = load_suite(root / "configs/benchmarks/mars_tracking_mastery_crossed_v1.json")
    assert suite["config"]["base_turn_progress_guard"] is True
    assert select_cases(suite, ["base"], "all") == select_cases(crossed, ["base"], "all")
    train = select_cases(suite, ["base"], "train", population="training")
    original = select_cases(crossed, ["base"], "train", population="training")
    assert len(train) == len({c["id"] for c in train}) == 45
    assert all(c in train for c in original)
    assert {c["turn_rad"] for c in train} == {-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06}
    report = {
        "complete": True,
        "controller": "task_pose_teacher",
        "suite_sha256": digest(suite),
        "source_hashes": source_hashes(),
        "cases": [
            {"id": c["id"], "mastered": True} for c in select_cases(crossed, ["base"], "all", population="admission")
        ],
    }
    with pytest.raises(ValueError):
        validate_preflight(report, suite, ["base"])
    report["cases"] = [
        {"id": c["id"], "mastered": True} for c in select_cases(suite, ["base"], "all", population="admission")
    ]
    validate_preflight(report, suite, ["base"])
