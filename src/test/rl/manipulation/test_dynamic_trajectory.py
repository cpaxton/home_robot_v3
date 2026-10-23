import numpy as np
import pytest

from emet.rl.manipulation.dynamic_trajectory import ArmTrajectory, admit, build, segment
from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.profiles import MARS


def test_replanning_preserves_position_velocity_acceleration_and_derivative_limits():
    q = np.array([0, -0.4, 0.2, 0.2, 0.0])
    path = build(q, q + 0.1, MARS, np.random.default_rng(7), replan_time=0.7, replan_goal=q - 0.08)
    first, second = path.segments
    before, after = first.sample(0.7), second.sample(0.7)
    for k in range(3):
        np.testing.assert_allclose(before[k], after[k], atol=1e-10)
    for time in np.linspace(0, path.end, 2001):
        _, v, a, j = path.sample(time)
        assert np.all(abs(v) <= np.asarray(MARS.speed) + 1e-6)
        assert np.all(abs(a) <= np.asarray(MARS.acceleration) + 1e-6)
        assert np.all(abs(j) <= 4 * np.asarray(MARS.acceleration) + 1e-6)
    hold = path.sample(path.end + 1)
    np.testing.assert_allclose(hold[0], q - 0.08, atol=1e-10)
    for derivative in hold[1:]:
        np.testing.assert_array_equal(derivative, np.zeros_like(q))


@pytest.mark.sim
def test_colliding_goal_is_rejected_without_moving_robot():
    env = ManipulationEnv(ManipulationConfig(robot="innate_mars", scenario="reach"))
    try:
        env.reset(seed=200014)
        qpos, timestamp = env.data.qpos.copy(), env.data.time
        env.control.base_mask[:] = False
        # One of the original full-distribution failures: shoulder intersects chassis.
        goal = np.array([0.1737303, -0.6819144, 0.5268412, 0.1003259, -0.1479153])
        path = ArmTrajectory([segment(env.control.positions(), goal, MARS.speed, MARS.acceleration)])
        result = admit(env, path)
        assert not result["accepted"]
        assert result["reason"] == "collision"
        assert result["contacts"]
        np.testing.assert_array_equal(env.data.qpos, qpos)
        assert env.data.time == timestamp
    finally:
        env.close()


@pytest.mark.sim
def test_dynamic_reset_is_reproducible_and_tracks_moving_reference():
    env = ManipulationEnv(
        ManipulationConfig(
            robot="innate_mars",
            scenario="reach",
            horizon=20,
            retarget_probability=1,
            dynamic_references=True,
            certify_reach_paths=True,
        )
    )
    try:
        env.reset(seed=300001)
        record = env.arm_trajectory.record()
        env.reset(seed=300001)
        assert env.arm_trajectory.record() == record
        assert env.trajectory_admission["accepted"]
        initial = env.control.reference.copy()
        for _ in range(35):
            _, _, done, truncated, info = env.step(np.zeros(env.action_space.shape))
            assert not (done or truncated), info
        assert env.retargets == 1
        assert np.linalg.norm(initial - env.control.reference) > 0.005
        for _ in range(365):
            _, _, done, truncated, info = env.step(np.zeros(env.action_space.shape))
            if done or truncated:
                break
        assert info["is_success"], info
        assert not info["unsafe_contact"]
        assert not info["wheel_lift"]
    finally:
        env.close()


def test_faster_checkpoint_cannot_hide_failures_or_worse_forces():
    from copy import deepcopy

    from emet.rl.manipulation.evaluation import paired_comparison

    row = {
        "fingerprint": "model",
        "initial": {"reference": [0]},
        "time": 5,
        "is_success": True,
        "initialization_failed": False,
        "unsafe_contact": False,
        "wheel_lift": False,
        "retention_lost": False,
    }
    summary = {
        "success_rate": 1,
        "joint_error_mean": 0.01,
        "ee_position_error_p95": 0.005,
        "ee_orientation_error_p95": 0.02,
        "base_goal_error_p95": 0.005,
        "peak_contact_force_p95": 1,
        "peak_contact_force_max": 1,
        "contact_impulse_p95": 1,
        "peak_measured_jerk_p95": 10,
        "peak_measured_jerk_max": 10,
    }
    teacher = {"episodes": [{**row, "episode_id": str(i)} for i in range(2)], "summary": {"reach": summary}}
    faster = deepcopy(teacher)
    for r in faster["episodes"]:
        r["time"] = 4
    assert paired_comparison(teacher, faster)["eligible"]
    failed = deepcopy(faster)
    failed["episodes"][1]["is_success"] = False
    failed["summary"]["reach"]["success_rate"] = 0.5
    assert not paired_comparison(teacher, failed)["eligible"]
    stronger = deepcopy(faster)
    stronger["summary"]["reach"]["peak_contact_force_p95"] = 1.01
    assert not paired_comparison(teacher, stronger)["eligible"]
    changed = deepcopy(faster)
    changed["episodes"][1]["initial"]["reference"] = [1]
    assert not paired_comparison(teacher, changed)["eligible"]
    base_teacher = deepcopy(teacher)
    base_teacher["summary"]["reach"].update(
        hand_target_count=0,
        ee_position_error_p95=None,
        ee_orientation_error_p95=None,
        joint_error_mean=None,
        base_contract_violation_p95=0,
    )
    base_candidate = deepcopy(base_teacher)
    assert paired_comparison(base_teacher, base_candidate)["eligible"]
    base_candidate["summary"]["reach"]["peak_contact_force_max"] = 2
    assert not paired_comparison(base_teacher, base_candidate)["eligible"]


@pytest.mark.sim
def test_clear_arm_path_can_be_rejected_for_base_compensation_sweep():
    env = ManipulationEnv(ManipulationConfig(robot="innate_mars", scenario="reach"))
    try:
        env.reset(seed=400003)
        goal = np.array([0.04201725, -0.65328251, 0.14466737, -0.11787113, -0.03407047])
        path = ArmTrajectory([segment(env.control.positions(), goal, MARS.speed, MARS.acceleration)])
        env.control.base_mask[:] = False
        assert admit(env, path)["accepted"]
        env.control.base_mask[:] = True
        env.control.hand_mask = True
        rejected = admit(env, path)
        assert not rejected["accepted"]
        assert rejected["reason"] == "base_compensation_collision"
    finally:
        env.close()
