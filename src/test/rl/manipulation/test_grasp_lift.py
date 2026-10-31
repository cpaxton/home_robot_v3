"""Loaded lift-off is a bounded carry subtype, never a reset teacher lift."""

from dataclasses import replace

import numpy as np
import pytest

from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.grasp_lift import SupportedGraspReadiness, prepare
from emet.rl.manipulation.retention_training import TRACKING_RETENTION_MIX, sample_carry_subtype


def config(**kwargs):
    return replace(
        ManipulationConfig(
            robot="innate_mars",
            scenario="carry",
            horizon=12,
            randomization_scale=0,
            action_mode="direct_velocity",
            trajectory_preview=True,
            tracking_reward=True,
            retention_context=True,
            balanced_grasp_setup=True,
            fixed_base_contract=True,
            grasp_lift_probability=1,
            contact_cone="elliptic",
            contact_impratio=3,
            support_half_width=0.04,
            wheel_feedback_gain=0.5,
            whole_body_settling=True,
            hand_position_tolerance=0.003,
            hand_orientation_tolerance=0.02,
            payload_mass_kg=0.05,
        ),
        **kwargs,
    )


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_invalid_lift_share_fails_closed(value):
    with pytest.raises(ValueError, match="grasp lift probability"):
        config(grasp_lift_probability=value)


@pytest.mark.parametrize(
    "change",
    [
        {"robot": "stretch"},
        {"balanced_grasp_setup": False},
        {"trajectory_preview": False},
        {"tracking_reward": False},
        {"retention_context": False},
        {"fixed_base_contract": False},
        {"action_mode": "residual"},
        {"tamp_carry_trajectories": True},
    ],
)
def test_lift_requires_complete_explicit_pose_and_safety_contract(change):
    with pytest.raises(ValueError):
        config(**change)


def test_lift_subtype_retains_tracking_majority_and_legacy_rng_at_zero():
    rng = np.random.default_rng(19)
    scenarios = rng.choice(["reach", "pregrasp", "carry", "place", "disturbance"], size=20000, p=TRACKING_RETENTION_MIX)
    modes = [sample_carry_subtype(s, rng, lift_probability=0.75, hold_probability=0.25) for s in scenarios]
    assert 0.085 < np.mean([m == "grasp_lift" for m in modes]) < 0.095
    assert 0.69 < np.mean(scenarios == "reach") < 0.71
    assert all(m is None for s, m in zip(scenarios, modes, strict=True) if s != "carry")
    a, b = np.random.default_rng(4), np.random.default_rng(4)
    for _ in range(20):
        expected = "hold" if b.random() < 0.25 else "ordinary"
        assert sample_carry_subtype("carry", a, lift_probability=0, hold_probability=0.25) == expected
    assert a.bit_generator.state == b.bit_generator.state


def test_contact_alone_does_not_admit_a_moving_or_unloaded_grasp():
    gate = SupportedGraspReadiness(0.05)
    pose = (np.zeros(3), np.eye(3))
    for _ in range(30):
        assert not gate.update(pose, bilateral=True, grip_velocity=0.02, supported=True)
    for _ in range(gate.window):
        accepted = gate.update(pose, bilateral=True, grip_velocity=0, supported=True)
    assert accepted
    assert not gate.update(pose, bilateral=True, grip_velocity=0, supported=False)
    assert gate.last["samples"] == 0
    for i in range(gate.window):
        moved = (np.array([i * 0.001, 0, 0]), np.eye(3))
        assert not gate.update(moved, bilateral=True, grip_velocity=0, supported=True)


@pytest.fixture
def lift_env():
    env = ManipulationEnv(config())
    try:
        obs, info = env.reset(seed=3100002)
        assert not info["initialization_failed"], env.grasp_lift_admission
        yield env, obs, info
    finally:
        env.close()


@pytest.mark.sim
def test_policy_starts_supported_with_closed_grip_and_full_pose_schedule(lift_env):
    env, obs, info = lift_env
    admission = env.grasp_lift_admission
    assert info["grasp_lift_trial"] and not info["carry_hold_trial"]
    assert env.fixed_base_trial and env.control.base_mask.all() and env.retention_reference is not None
    assert admission["initial_support_load_N"] > 0.1 * env.model.body_mass[env.payload] * 9.81
    assert abs(env.data.xpos[env.payload, 2] - env.initial_object_height) < 0.003
    assert 0.048 <= admission["requested_lift_m"] <= 0.06
    assert env.control.phase == "carry" and env.control.grip_reference == env.profile.gripper_closed
    assert not env.control.release_authorized and set(obs) == {"history", "preview"}
    assert admission["minimum_sampled_margin_m"] >= 0.005
    assert env.arm_trajectory.end + 2 <= env.config.horizon
    assert env.control.base_yaw_deadband < env.active_base_heading_tolerance


@pytest.mark.sim
def test_preparing_targets_never_moves_physics_and_obeys_native_derivatives(lift_env):
    env, _, _ = lift_env
    before = [x.copy() for x in (env.data.qpos, env.data.qvel, env.data.ctrl)]
    time = float(env.data.time)
    prepare(env)
    assert not env.initialization_failed, env.grasp_lift_admission
    for actual, expected in zip((env.data.qpos, env.data.qvel, env.data.ctrl), before, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert env.data.time == time
    for t in np.linspace(0, env.arm_trajectory.end, 101):
        _, v, a, j = env.arm_trajectory.sample(t)
        assert np.all(abs(v) <= np.asarray(env.profile.speed) * 0.5 + 1e-8)
        assert np.all(abs(a) <= np.asarray(env.profile.acceleration) * 0.5 + 1e-8)
        assert np.isfinite(j).all()


@pytest.mark.sim
def test_measured_collision_stops_and_clears_departure_permission(lift_env, monkeypatch):
    env, _, _ = lift_env
    guard = env.grasp_lift_guard
    monkeypatch.setattr(guard, "collides", lambda data: True)
    _, _, done, truncated, info = env.step(np.zeros(env.profile.action_size))
    assert done and not truncated and info["code"] == "grasp_lift_collision"
    assert env.control.stopped and guard.context is None and not guard.checker._support_departure


@pytest.mark.sim
def test_lift_payload_selection_does_not_replace_other_tasks_object_mix():
    cfg = config(grasp_lift_payload_object="object1")
    env = ManipulationEnv(cfg)
    try:
        _, info = env.reset(seed=3101000)
        assert not info["initialization_failed"] and env.payload_name == "object1"
    finally:
        env.close()
    env = ManipulationEnv(replace(cfg, scenario="reach", grasp_lift_probability=0))
    try:
        names = set()
        for seed in range(10, 20):
            env.reset(seed=seed)
            assert not env.grasp_lift_trial
            names.add(env.payload_name)
        assert names == {"object1", "object2"}
    finally:
        env.close()


@pytest.mark.sim
@pytest.mark.parametrize("phase,index", [("start", 0), ("mid", 1), ("late", 2), ("hold", 3)])
def test_physical_phase_prefix_retains_grasp_contract_and_correct_reference_clock(phase, index):
    import json
    from pathlib import Path

    cfg = ManipulationConfig(
        **json.loads(Path("configs/benchmarks/mars_grasp_lift_curriculum_v1.json").read_text())["environment"]
    )
    weights = [0.0, 0.0, 0.0, 0.0]
    weights[index] = 1
    cfg = replace(
        cfg,
        scenario="carry",
        grasp_lift_probability=1,
        jerk_limited_lift_reference=True,
        grasp_lift_phase_weights=weights,
    )
    env = ManipulationEnv(cfg)
    try:
        _, info = env.reset(seed=3101000)
        assert not info["initialization_failed"], env.grasp_lift_admission
        assert info["time"] == 0 and info["grasp_lift_start_mode"] == phase
        record = env.grasp_lift_admission
        assert record["phase_reset"]["prefix_failure"] is None if phase != "start" else env.reference_time_offset == 0
        assert env.control.action_mode == "direct_velocity" and not env.control.release_authorized
        assert env._reference_end_elapsed() == pytest.approx(env.reference_time_offset - env.arm_trajectory.end)
        if phase != "start":
            assert env.reference_time_offset > 0
            assert record["supported_grasp_initial_state"] != record["initial_state"]
            assert record["phase_reset"]["prefix_force_N"] < env.profile.unsafe_force
            np.testing.assert_allclose(env.last_velocity, env.control.velocities())
            assert env.control.history[-1].tolist() != env.control.history[0].tolist()
        if phase == "hold":
            assert env._reference_complete()
            assert env.data.xpos[env.payload, 2] - env.initial_object_height > 0.04
        env.step(np.zeros(env.profile.action_size))
        assert (
            env.start_time + env.reference_time_offset != env.start_time
            if phase != "start"
            else env.reference_time_offset == 0
        )
    finally:
        env.close()


@pytest.mark.sim
def test_failed_prefix_is_rejected_before_policy_and_cannot_rebase_retention(lift_env, monkeypatch):
    from emet.rl.manipulation.grasp_lift import prepare_phase_reset

    env, _, _ = lift_env
    env.config = replace(env.config, jerk_limited_lift_reference=True, grasp_lift_phase_weights=(0, 1, 0, 0))
    initial_reference = env.retention_reference
    original = env._tick

    def faulted(*args, **kwargs):
        evidence = original(*args, **kwargs)
        env.monitor.wheel_lift = True
        return evidence

    monkeypatch.setattr(env, "_tick", faulted)
    prepare_phase_reset(env)
    assert env.initialization_failed and env.initialization_failure_reason == "unsafe_physical_lift_prefix"
    assert env.control.stopped and env.grasp_lift_guard.context is None
    assert env.retention_reference is initial_reference
