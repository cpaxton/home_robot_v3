"""Carry admission, physical execution and benchmark qualification contracts."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from emet.rl.manipulation.carry_benchmark import (
    CANONICAL_SUITE,
    canonical_digests,
    case_config,
    default_config,
    digest,
    make_suite,
    ready,
    require_training_readiness,
    source_hashes,
)
from emet.rl.manipulation.carry_training import prepare
from emet.rl.manipulation.env import ManipulationEnv, sample_carry_events


def test_smoke_suite_cannot_qualify_training():
    assert len(make_suite()["cases"]) == 300
    assert len({c["id"] for c in make_suite()["cases"]}) == 300
    assert not ready({"complete": False, "controller": "teacher"})
    assert not ready({"complete": True, "controller": "teacher", "episodes": []})


def test_training_mixture_contains_quiet_and_disturbed_pose_categories():
    config = replace(default_config(), carry_target_mode="mixed", carry_push_scale=0.02, carry_push_probability=0.25)
    rng = np.random.default_rng(31)
    events = [sample_carry_events(config, rng, "carry") for _ in range(2000)]
    assert {mode for mode, _ in events} == {"translation", "orientation", "combined"}
    assert 0.2 < np.mean([push for _, push in events]) < 0.3
    for probability in (0, 1):
        fixed = replace(config, carry_push_probability=probability)
        assert all(sample_carry_events(fixed, rng, "carry")[1] == bool(probability) for _ in range(20))


@pytest.mark.sim
def test_frozen_disturbance_uses_wider_support_without_weakening_the_push():
    env = ManipulationEnv(replace(default_config(), carry_push_scale=0.1))
    try:
        _, info = env.reset(seed=1904001)
        assert not info["initialization_failed"]
        setup = info["randomization"]["grasp_setup"]
        assert setup["selection_reason"] == "maximum_margin_for_disturbance"
        assert setup["minimum_sampled_margin_m"] > 0.02
        np.testing.assert_allclose(
            env.disturbance_wrench,
            [
                0.3225620618129116,
                -0.19520660592966363,
                0.06488432700883441,
                -0.0069418701320262555,
                0.012867630979818892,
                -0.006184681858643481,
            ],
            atol=1e-12,
        )
        for _ in range(round(env.config.horizon / env.profile.dt) + 1):
            _, _, done, truncated, info = env.step(env.teacher_action())
            if done or truncated:
                break
        assert info["is_success"], info
        assert not info["wheel_lift"]
    finally:
        env.close()


def passing_evidence():
    suite = json.loads(CANONICAL_SUITE.read_text())
    hashes = source_hashes()
    rows = [
        {
            "case_id": c["id"],
            "seed": c["seed"],
            "benchmark_config": asdict(case_config(suite, c)),
            "payload_mass_kg": c["payload_mass_kg"],
            "stratum": c["stratum"],
            "is_success": True,
            "initialization_failed": False,
            "unsafe_contact": False,
            "wheel_lift": False,
            "retention_lost": False,
            "initial": {"rejected_candidates": []},
            "fingerprint": "model",
            "code": "ok",
        }
        for c in suite["cases"]
    ]
    carry = {
        "complete": True,
        "controller": "teacher",
        "episodes": rows,
        "suite": suite,
        "suite_sha256": digest(suite),
        "source_hashes": hashes,
    }
    pick = {
        "regressions_sha256": canonical_digests()["expected_regressions_sha256"],
        "complete": True,
        "cases": [
            {
                "seed": seed,
                "requested_payload_mass_kg": mass,
                "success": True,
                "stages": [],
                "control_source_sha256": digest(hashes),
                "controller": "velocity_teacher",
                "fingerprint": "model",
            }
            for seed in range(1860001, 1860011)
            for mass in (0.02, 0.05, 0.1)
        ],
    }
    return carry, pick


def assert_ready(carry, pick):
    return require_training_readiness(carry, pick, default_config(), "model", **canonical_digests())


def test_training_requires_complete_current_safe_evidence():
    carry, pick = passing_evidence()
    rows = carry["episodes"]
    assert_ready(carry, pick)
    rows[0]["wheel_lift"] = True
    with pytest.raises(ValueError, match="readiness"):
        assert_ready(carry, pick)
    rows[0]["wheel_lift"] = False
    pick["cases"][0]["success"] = False
    with pytest.raises(ValueError, match="pick/place"):
        assert_ready(carry, pick)
    pick["cases"][0]["success"] = True
    carry["source_hashes"] = {}
    with pytest.raises(ValueError, match="stale"):
        assert_ready(carry, pick)


@pytest.mark.sim
def test_open_aperture_overshoot_is_not_misclassified_as_closed_and_fault_clears_wheel_bias():
    env = ManipulationEnv(default_config())
    try:
        env.reset(seed=1850002)
        c = env.control
        env.data.qpos[c.grip_q] = env.profile.gripper_open + 0.007
        assert c.gripper_is_open()
        env.data.qpos[c.grip_q] = env.profile.gripper_open - 0.01
        assert not c.gripper_is_open()
        c.base_velocity_integral[:] = 0.01
        c.stop()
        assert not c.base_bias_compensation_enabled
        np.testing.assert_array_equal(c.base_velocity_integral, 0)
        with pytest.raises(ValueError):
            c.set_base_bias_compensation(True)
    finally:
        env.close()


@pytest.mark.sim
@pytest.mark.parametrize("seed", [1850001, 1850002, 1850007])
def test_frozen_carry_is_admitted_and_planning_does_not_edit_live_physics(seed):
    env = ManipulationEnv(default_config())
    try:
        obs, info = env.reset(seed=seed)
        assert not info["initialization_failed"], info
        assert info["trajectory_admission"]["accepted"]
        assert info["trajectory_admission"]["minimum_sampled_margin_m"] >= 0.005
        assert obs["preview"].shape[1] == 18
        before = [
            a.copy() for a in (env.data.qpos, env.data.qvel, env.data.ctrl, env.model.body_pos, env.model.geom_size)
        ]
        time = env.data.time
        prepare(env)
        assert not env.initialization_failed
        for a, b in zip(
            before, (env.data.qpos, env.data.qvel, env.data.ctrl, env.model.body_pos, env.model.geom_size), strict=True
        ):
            np.testing.assert_array_equal(a, b)
        assert time == env.data.time
        assert env.arm_trajectory.end + 2 <= env.config.horizon
        for t in np.linspace(0, env.arm_trajectory.end, 101):
            _, v, a, j = env.arm_trajectory.sample(t)
            assert np.all(abs(v) <= 0.5 * np.array(env.profile.speed) + 1e-8)
            assert np.all(abs(a) <= 0.5 * np.array(env.profile.acceleration) + 1e-8)
            assert np.all(abs(j) <= 2 * np.array(env.profile.acceleration) + 1e-8)
    finally:
        env.close()


@pytest.mark.sim
def test_carry_retarget_is_continuous_and_does_not_leak_before_event():
    env = ManipulationEnv(replace(default_config(), retarget_probability=1))
    try:
        env.reset(seed=1850002)
        assert not env.initialization_failed
        path = env.arm_trajectory
        t = path.replan_time
        assert t is not None
        left, right = path.segments
        for a, b in zip(left.sample(t)[:3], right.sample(t)[:3], strict=True):
            np.testing.assert_allclose(a, b, atol=1e-8)
        # Until announced, even distant preview samples keep the first goal.
        before = env._sample_task_reference(env.start_time + path.end + 1)[1]
        env.retargets = 1
        after = env._sample_task_reference(env.start_time + path.end + 1)[1]
        assert not np.allclose(before, after)
    finally:
        env.close()


@pytest.mark.sim
def test_late_disturbance_reserves_recovery_and_can_finish_contract():
    env = ManipulationEnv(replace(default_config(), payload_mass_kg=0.02, carry_push_scale=0.1))
    try:
        env.reset(seed=1904010)
        assert not env.initialization_failed
        end = env.disturbance_start + env.disturbance_steps
        assert end * env.profile.dt <= env.config.horizon - 2
        for _ in range(round(env.config.horizon / env.profile.dt) + 1):
            _, _, done, truncated, info = env.step(env.teacher_action())
            if done or truncated:
                break
        assert info["is_success"], info
    finally:
        env.close()


def test_same_shape_altered_catalog_and_case_parameters_are_rejected():
    carry, pick = passing_evidence()
    carry["suite"]["config"]["horizon"] += 1
    carry["suite_sha256"] = digest(carry["suite"])
    with pytest.raises(ValueError, match="canonical benchmark"):
        assert_ready(carry, pick)
    carry, pick = passing_evidence()
    pick["regressions_sha256"] = "other-catalog"
    with pytest.raises(ValueError, match="catalog checksum"):
        assert_ready(carry, pick)
    carry, pick = passing_evidence()
    carry["episodes"][0]["seed"] += 1
    with pytest.raises(ValueError, match="case parameters"):
        assert_ready(carry, pick)


@pytest.mark.parametrize("module", ["rl/control.py", "rl/contact.py", "rl/pose_targets.py"])
def test_source_manifest_pins_shared_safety_modules_and_detects_changes(module, monkeypatch):
    before = source_hashes()
    assert {
        "rl/control.py",
        "rl/contact.py",
        "rl/pose_targets.py",
        "eval/physical_tamp.py",
        "motion/arm_rrt.py",
        "rl/regression_gate.py",
    } <= before.keys()
    original = Path.read_bytes

    def edited(path):
        data = original(path)
        return data + b"\n# safety change\n" if str(path).endswith("/emet/" + module) else data

    monkeypatch.setattr(Path, "read_bytes", edited)
    after = source_hashes()
    assert {key for key in before if before[key] != after[key]} == {module}


def test_per_mass_gate_is_not_a_per_stratum_or_end_to_end_claim():
    carry, _ = passing_evidence()
    # 91/95 admitted successes passes 95% conditional, but is 91% end-to-end.
    for row in carry["episodes"][:5]:
        row["initialization_failed"] = True
        row["is_success"] = False
    for row in carry["episodes"][5:9]:
        row["is_success"] = False
    assert ready(carry)
    from emet.rl.manipulation.carry_benchmark import summarize

    assert summarize(carry["episodes"])["0.02"]["end_to_end_success_rate"] == 0.91


def test_training_cannot_relax_the_qualified_pose_contract():
    carry, pick = passing_evidence()
    with pytest.raises(ValueError, match="hand_position_tolerance"):
        require_training_readiness(
            carry, pick, replace(default_config(), hand_position_tolerance=0.015), "model", **canonical_digests()
        )
