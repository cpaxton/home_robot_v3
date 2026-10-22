from copy import deepcopy

import numpy as np
import pytest

from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.evaluation import curriculum_ready, summarize, target_coverage_ready


def test_endpoint_diagnostics_do_not_count_accurate_moving_hand_as_success():
    row = {
        "scenario": "reach",
        "initialization_failed": False,
        "initialization_rejections": 0,
        "is_success": False,
        "retention_lost": False,
        "unsafe_contact": False,
        "wheel_lift": False,
        "hand_target_active": True,
        "ee_position_error": 0.004,
        "ee_orientation_error": 0.04,
        "base_linear_velocity_m_s": 0.01,
        "base_yaw_velocity_rad_s": 0,
        "config": {"hand_position_tolerance": 0.005, "hand_orientation_tolerance": 0.05},
    }
    metrics = summarize([row])["reach"]
    assert metrics["success_rate"] == 0
    mixed = summarize([row, {**row, "hand_target_active": False}])["reach"]
    assert mixed["hand_target_count"] == 1
    assert mixed["endpoint_hand_within_rate"] == 1
    assert mixed["endpoint_base_moving_count"] == 2
    assert mixed["endpoint_hand_position_outside_count"] == 0
    assert mixed["endpoint_hand_orientation_outside_count"] == 0
    assert mixed["endpoint_hand_within_10mm_0.1rad_rate"] == 1
    never_held = summarize([{**row, "peak_hold_base_drift_m": 0, "time_to_first_settle": None}])["reach"]
    assert never_held["peak_hold_base_drift_m_p95"] is None
    held = summarize([{**row, "peak_hold_base_drift_m": 0.003, "time_to_first_settle": 2}])["reach"]
    assert held["peak_hold_base_drift_m_p95"] == 0.003
    inactive = summarize([{**row, "hand_target_active": False}])["reach"]
    assert inactive["endpoint_hand_within_rate"] is None
    assert inactive["endpoint_hand_position_outside_count"] is None
    failed = summarize([{**row, "ee_position_error": 0.015, "ee_orientation_error": 0.08, "code": "tracking_timeout"}])[
        "reach"
    ]
    assert failed["endpoint_hand_position_outside_count"] == 1
    assert failed["endpoint_hand_orientation_outside_count"] == 1
    assert failed["endpoint_hand_within_10mm_0.1rad_rate"] == 0
    assert failed["endpoint_hand_within_20mm_0.1rad_rate"] == 1
    assert failed["terminal_tracking_timeout_count"] == 1
    assert metrics["endpoint_hand_within_rate"] == 1
    assert metrics["endpoint_base_still_rate"] == 0
    assert metrics["endpoint_hand_within_but_base_moving_rate"] == 1
    metrics = summarize([{**row, "base_linear_velocity_m_s": 0}])["reach"]
    assert metrics["endpoint_hand_within_but_base_moving_rate"] == 0
    assert metrics["success_rate"] == 0


def test_reach_coverage_can_expand_before_strict_success_without_relaxing_safety():
    group = {
        "success_rate": 0,
        "initialization_failure_rate": 0,
        "unsafe_contact_rate": 0,
        "wheel_lift_rate": 0,
        "retention_lost_rate": 0,
        "ee_position_error_p95": 0.02,
        "ee_orientation_error_p95": 0.1,
        "base_contract_violation_p95": 0.01,
    }
    assert target_coverage_ready({"reach": group}, 20)
    assert not curriculum_ready({"reach": group}, 20)
    assert not target_coverage_ready({"reach": group}, 19)
    assert not target_coverage_ready({"carry": group}, 20)
    assert not target_coverage_ready({}, 20)
    for key, value in [
        ("wheel_lift_rate", 0.05),
        ("unsafe_contact_rate", 0.05),
        ("retention_lost_rate", 0.05),
        ("initialization_failure_rate", 0.05),
        ("ee_position_error_p95", 0.03),
        ("ee_orientation_error_p95", 0.16),
        ("base_contract_violation_p95", 0.03),
        ("ee_position_error_p95", None),
    ]:
        assert not target_coverage_ready({"reach": {**group, key: value}}, 20)


def test_curriculum_requires_sample_size_and_every_category_to_pass():
    good = {
        "success_rate": 0.95,
        "initialization_failure_rate": 0,
        "unsafe_contact_rate": 0,
        "wheel_lift_rate": 0,
        "retention_lost_rate": 0,
    }
    summary = {"reach": good, "carry": deepcopy(good)}
    assert curriculum_ready(summary, 20)
    assert not curriculum_ready(summary, 19)
    assert not curriculum_ready({}, 20)
    for key, value in [
        ("success_rate", 0.89),
        ("initialization_failure_rate", 0.01),
        ("unsafe_contact_rate", 0.01),
        ("wheel_lift_rate", 0.01),
        ("retention_lost_rate", 0.01),
        ("wheel_lift_rate", None),
    ]:
        failed = deepcopy(summary)
        failed["carry"][key] = value
        assert not curriculum_ready(failed, 20)


@pytest.mark.sim
def test_mars_carry_resets_physically_lift_both_payload_shapes_without_attachment():
    env = ManipulationEnv(ManipulationConfig(robot="innate_mars", scenario="carry", initialization_attempts=3))
    shapes = set()
    try:
        for seed in (710001, 710003):
            _, info = env.reset(seed=seed)
            assert not info["initialization_failed"], info
            assert env.data.xpos[env.payload, 2] - env.initial_object_height > 0.04
            assert env.monitor.update(env.data, env.payload)["bilateral_contact"]
            assert not env.control.stopped
            assert not env.monitor.wheel_lift
            assert env.monitor.peak_force < env.profile.unsafe_force
            assert all(int(t) == 2 for t in env.model.eq_type)  # Joint mimic only; no payload weld.
            shapes.add(env.payload_name)
        assert shapes == {"object1", "object2"}
    finally:
        env.close()


@pytest.mark.sim
def test_randomization_zero_keeps_nominal_mass_friction_and_delay():
    env = ManipulationEnv(ManipulationConfig(robot="innate_mars", scenario="reach", randomization_scale=0))
    try:
        env.reset(seed=19)
        assert env.command_delay == 0
        np.testing.assert_allclose(env.model.body_mass, env.mass)
        np.testing.assert_allclose(env.model.geom_friction, env.friction)
    finally:
        env.close()


@pytest.mark.sim
def test_training_retarget_and_push_are_seeded_and_do_not_teleport():
    config = ManipulationConfig(
        robot="innate_mars",
        scenario="reach",
        randomization_scale=0,
        target_scale=0.1,
        retarget_probability=1,
        reach_push_probability=1,
    )
    env = ManipulationEnv(config)
    try:
        env.reset(seed=17)
        goal = tuple(v.copy() for v in env.retarget_goal)
        scheduled = env.retarget_step
        env.reset(seed=17)
        assert env.retarget_step == scheduled
        for actual, expected in zip(env.retarget_goal, goal, strict=True):
            np.testing.assert_array_equal(actual, expected)
        tick = env._tick
        integrated = []

        def capture_integrated_state(action, **kwargs):
            result = tick(action, **kwargs)
            integrated.append(env.data.qpos.copy())
            return result

        env._tick = capture_integrated_state
        for _ in range(scheduled):
            _, _, done, truncated, info = env.step(np.zeros(env.action_space.shape))
            assert not (done or truncated), info
            np.testing.assert_array_equal(env.data.qpos, integrated[-1])
        assert env.retargets == 1
        np.testing.assert_allclose(env.control.base_goal, goal[0])
        np.testing.assert_allclose(env.control.reference, goal[1])
        np.testing.assert_allclose(env.control.hand_goal, goal[2])
        assert env.disturbance_start is not None
        assert not np.any(env.disturbance_wrench)  # Zero randomization means no disturbance.
    finally:
        env.close()
