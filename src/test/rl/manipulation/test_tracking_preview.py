"""Preview, reward, and continuing-episode contracts for task-space tracking."""

import mujoco
import numpy as np
import pytest

from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.evaluation import tracking_precision_ready
from emet.rl.manipulation.tracking import (
    PREVIEW_OFFSETS,
    PREVIEW_WIDTH,
    TaskFrameFK,
    endpoint_motion_cost,
    stationary_tracking_score,
    tracking_score,
)


def test_endpoint_motion_ramp_is_bounded_and_resets_with_new_reference_time():
    def cost(elapsed, arm=(0.03,), base=(0.005, 0.02), hand=True):
        return endpoint_motion_cost(elapsed, arm, base, hand_active=hand)

    assert cost(-1) == cost(0) == 0
    assert cost(1) == pytest.approx(0.5 * cost(2))
    assert cost(20) == cost(2)
    assert cost(2, arm=(0,), base=(0, 0)) == 0
    assert 0 < cost(2) < cost(2, arm=(10,), base=(10, 10)) < 2
    assert cost(2, hand=False) < cost(2)
    assert cost(-3) == 0  # Newly installed reference is still in progress.
    with pytest.raises(ValueError):
        ManipulationConfig(endpoint_motion_weight=float("nan"))


def test_stationary_reward_prefers_accurate_quiet_hold_without_a_threshold_jump():
    def score(pos=0, angle=0, arm=(0,), base=(0, 0), hand=True):
        return stationary_tracking_score(pos, angle, arm, base, hand_active=hand)

    assert score() == 1
    assert score(pos=0.02) < score(pos=0.005) < score()
    assert score(angle=0.1) < score(angle=0.05) < score()
    assert score(base=(0.01, 0)) < score(base=(0.005, 0)) < score()
    assert score(base=(0, 0.04)) < score(base=(0, 0.02)) < score()
    assert score(arm=(0.06,)) < score(arm=(0.03,)) < score()
    assert score(pos=1, angle=1, arm=(1,), hand=False) == 1
    assert abs(score(pos=0.004999) - score(pos=0.005001)) < 0.001


def test_coupled_reward_requires_both_position_and_orientation_and_tightens():
    assert tracking_score(0, 0, 0) == 1
    assert tracking_score(0.01, 0, 0) < 1
    assert tracking_score(0, 0.1, 0) < 1
    assert tracking_score(0.01, 0.1, 0) < tracking_score(0.01, 0, 0)
    assert tracking_score(0.01, 0.1, 4) < tracking_score(0.01, 0.1, 0)
    good = {
        "initialization_failure_rate": 0,
        "unsafe_contact_rate": 0,
        "wheel_lift_rate": 0,
        "retention_lost_rate": 0,
        "ee_position_error_p95": 0.02,
        "ee_orientation_error_p95": 0.1,
        "base_contract_violation_p95": 0.01,
    }
    assert tracking_precision_ready({"reach": good}, 0, 20)
    assert not tracking_precision_ready({"reach": good}, 0, 19)
    assert not tracking_precision_ready({"reach": good}, 4, 20)
    assert not tracking_precision_ready({"reach": {**good, "wheel_lift_rate": 0.05}}, 0, 20)
    assert not tracking_precision_ready({"reach": {**good, "ee_position_error_p95": None}}, 0, 20)


@pytest.mark.sim
@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
def test_task_frame_fk_and_preview_do_not_edit_physics_or_follow_base_disturbance(robot):
    e = ManipulationEnv(
        ManipulationConfig(robot=robot, scenario="reach", trajectory_preview=True, randomization_scale=0)
    )
    try:
        obs, _ = e.reset(seed=2100001)
        assert e.observation_space.contains(obs)
        assert obs["preview"].shape == (len(PREVIEW_OFFSETS), PREVIEW_WIDTH)
        np.testing.assert_allclose(obs["preview"][:, 0], PREVIEW_OFFSETS)
        fk = TaskFrameFK(e.model, e.data, e.control)
        base, q = e.control.base_pose().copy(), e.control.positions().copy()
        expected = fk(base, q)
        np.testing.assert_allclose(expected, e.control.ee_pose(), atol=1e-10)
        state = e.data.qpos.copy(), e.data.qvel.copy(), float(e.data.time)
        for _ in range(3):
            e.control.observe()
            fk(base, q + 0.01)
        np.testing.assert_array_equal(e.data.qpos, state[0])
        np.testing.assert_array_equal(e.data.qvel, state[1])
        assert e.data.time == state[2]
        # This is a unit-test perturbation, not a scored collision recovery.
        address = e.model.jnt_qposadr[e.model.body_jntadr[e.control.base]]
        e.data.qpos[address] += 0.02
        mujoco.mj_forward(e.model, e.data)
        np.testing.assert_allclose(fk(base, q), expected, atol=1e-10)
    finally:
        e.close()


@pytest.mark.sim
def test_preview_hides_unannounced_retarget_but_exposes_announced_path():
    e = ManipulationEnv(
        ManipulationConfig(
            robot="innate_mars",
            scenario="reach",
            trajectory_preview=True,
            dynamic_references=True,
            retarget_probability=1,
            randomization_scale=0,
            target_scale=0.25,
            horizon=20,
        )
    )
    try:
        e.reset(seed=2100001)
        assert not e.initialization_failed
        assert len(e.arm_trajectory.segments) == 2
        e.control.hand_mask = True
        before = e.control.observe()["preview"].copy()
        second = e.arm_trajectory.segments[1]
        second.coefficients[0, 1] += 0.1
        hidden = e.control.observe()["preview"]
        np.testing.assert_array_equal(hidden, before)
        e.retargets = 1
        announced = e.control.observe()["preview"]
        assert not np.allclose(announced, before)
        assert np.isfinite(announced).all()
    finally:
        e.close()


@pytest.mark.sim
def test_preview_masks_targets_without_locking_unconstrained_base():
    e = ManipulationEnv(
        ManipulationConfig(
            robot="innate_mars",
            scenario="reach",
            trajectory_preview=True,
            action_mode="direct_velocity",
            randomization_scale=0,
        )
    )
    try:
        e.reset(seed=2100002)
        c = e.control
        c.base_mask[:] = False
        c.hand_mask = True
        c.base_command[:] = 0
        action = np.zeros(e.action_space.shape)
        action[0] = 1
        c.accept(action)
        assert c.base_command[0] > 0  # Base is available to meet the hand task.
        preview = c.observe()["preview"]
        np.testing.assert_array_equal(preview[:, 1:7], 0)
        c.hand_mask = False
        c.base_mask[:] = True
        c.accept(action)
        np.testing.assert_array_equal(c.observe()["preview"][:, 7:17], 0)
        c.preview_enabled = False
        c.base_mask[:] = False
        c.base_command[:] = 0
        c.accept(action)
        assert c.base_command[0] == 0  # Legacy bundle mask semantics are retained.
    finally:
        e.close()


@pytest.mark.sim
def test_tracking_success_keeps_controller_live_until_horizon_and_bonus_is_once():
    e = ManipulationEnv(
        ManipulationConfig(
            robot="innate_mars",
            scenario="reach",
            trajectory_preview=True,
            tracking_reward=True,
            action_mode="direct_velocity",
            randomization_scale=0,
            horizon=3,
        )
    )
    try:
        e.reset(seed=2100002)
        c = e.control
        c.set_reference(c.positions())
        c.set_pose_targets(c.base_pose(), np.zeros(3, dtype=bool), c.ee_pose(), True)
        e.arm_trajectory = None
        rewards = []
        nonterminal_success = False
        for _ in range(61):
            _, reward, done, truncated, info = e.step(e.teacher_action())
            rewards.append(reward)
            if info["is_success"] and not truncated:
                assert not done and not c.stopped
                nonterminal_success = True
            if done or truncated:
                break
        assert nonterminal_success
        assert truncated and not done and c.stopped
        assert info["time_to_first_settle"] is not None
        assert sum(r > 5 for r in rewards) == 1
        assert info["is_success"]
    finally:
        e.close()


def test_task_pose_trajectory_interpolates_rotations_and_rejects_bad_inputs():
    from scipy.spatial.transform import Rotation

    from emet.rl.manipulation.tracking import TaskPoseTrajectory

    rotations = Rotation.from_euler("z", [3.1, -3.1]).as_matrix()
    hands = np.concatenate([np.array([[0, 0, 0], [0.02, 0, 0]]), rotations[:, :, :2].reshape(2, 6)], axis=1)
    trajectory = TaskPoseTrajectory([0, 1], [[0, 0, 3.1], [0.1, 0, -3.1]], hands, [0.2, 0.3], base_mask=[1, 1, 1])
    base, hand, grip = trajectory.sample(0.5)
    assert abs(abs(base[2]) - np.pi) < 1e-8
    assert base[0] == pytest.approx(0.05)
    assert hand[0] == pytest.approx(0.01)
    assert hand[3] == pytest.approx(-1)
    assert grip == pytest.approx(0.25)
    with pytest.raises(ValueError):
        TaskPoseTrajectory([0, 0], trajectory.base, hands, [0.2, 0.3], base_mask=[1, 1, 1])
    with pytest.raises(ValueError):
        TaskPoseTrajectory([0, 1], trajectory.base, np.zeros((2, 9)), [0.2, 0.3], base_mask=[1, 1, 1])


@pytest.mark.sim
def test_pose_schedule_runs_without_ik_or_joint_reference_hint_and_cannot_release_or_clear_fault(monkeypatch):
    from emet.rl.manipulation.tracking import TaskPoseTrajectory

    e = ManipulationEnv(
        ManipulationConfig(
            robot="innate_mars",
            scenario="reach",
            trajectory_preview=True,
            tracking_reward=True,
            action_mode="direct_velocity",
            randomization_scale=0,
            horizon=3,
        )
    )
    try:
        e.reset(seed=2100002)
        c = e.control
        base = c.base_pose().copy()
        start = c.ee_pose().copy()
        goal = start.copy()
        goal[0] += 0.01
        trajectory = TaskPoseTrajectory(
            [0, 1],
            np.repeat(base[None], 2, axis=0),
            [start, goal],
            [e.profile.gripper_closed, e.profile.gripper_open],
            base_mask=[1, 1, 1],
        )

        def no_ik(*args, **kwargs):
            raise AssertionError("runtime pose commands must not invoke IK")

        monkeypatch.setattr(e, "_vertical_reference", no_ik)
        state = e.data.qpos.copy(), e.data.qvel.copy(), float(e.data.time)
        history = c.history.copy()
        c.grip_reference = e.profile.gripper_closed
        e.set_task_trajectory(trajectory)
        assert e._reference_end_elapsed() == pytest.approx(-1)
        np.testing.assert_array_equal(e.data.qpos, state[0])
        np.testing.assert_array_equal(e.data.qvel, state[1])
        assert e.data.time == state[2]
        np.testing.assert_array_equal(c.history, history)
        obs = c.observe()
        np.testing.assert_array_equal(obs["history"][-1, 20:25], 0)
        assert not c.release_authorized
        assert c.grip_reference == e.profile.gripper_closed
        for _ in range(10):
            e.step(np.zeros(e.action_space.shape))
        assert c.hand_goal[0] == pytest.approx(start[0] + 0.005, abs=1e-7)
        assert e._reference_end_elapsed() == pytest.approx(-0.5)
        e.set_task_trajectory(trajectory)
        assert e._reference_end_elapsed() == pytest.approx(-1)
        assert not c.release_authorized
        c.stop()
        with pytest.raises(ValueError):
            e.set_task_trajectory(trajectory)
        assert c.stopped
        e.reset(seed=2100002)
        assert c.task_trajectory is None
        assert c.preview_sampler == e._sample_task_reference
    finally:
        e.close()


@pytest.mark.sim
def test_gt_tamp_path_can_be_retimed_into_task_space_without_moving_the_robot():
    from emet.rl.manipulation.tamp_cases import plan_case, task_pose_trajectory

    e = ManipulationEnv(ManipulationConfig(robot="innate_mars", scenario="reach", randomization_scale=0))
    try:
        e.reset(seed=2100002)
        q = e.control.positions().copy()
        q[2] += 0.02
        case = plan_case(e, q)
        assert case["planning_error"] is None
        state = e.data.qpos.copy(), e.data.qvel.copy(), float(e.data.time)
        trajectory = task_pose_trajectory(e, case)
        assert trajectory.end > 0
        np.testing.assert_allclose(trajectory.sample(0)[1], e.control.ee_pose(), atol=1e-9)
        np.testing.assert_allclose(
            trajectory.sample(trajectory.end)[1], e._fk_hand_pose(e.control.base_pose(), q), atol=1e-9
        )
        np.testing.assert_array_equal(e.data.qpos, state[0])
        np.testing.assert_array_equal(e.data.qvel, state[1])
        assert e.data.time == state[2]
    finally:
        e.close()


@pytest.mark.sim
def test_hand_only_distribution_uses_whole_body_fk_without_leaking_a_base_solution():
    e = ManipulationEnv(
        ManipulationConfig(
            robot="innate_mars",
            scenario="reach",
            trajectory_preview=True,
            tracking_reward=True,
            dynamic_references=True,
            certify_reach_paths=True,
            randomization_scale=0,
            target_scale=0.25,
            horizon=20,
        )
    )
    try:
        for seed in range(2100000, 2100010):
            obs, _ = e.reset(seed=seed)
            if e.control.hand_mask and not e.control.base_mask.any():
                break
        else:
            raise AssertionError("test seeds must include an unconstrained-base hand task")
        c = e.control
        assert np.linalg.norm(c.base_goal - c.base_pose()) > 1e-4
        arm_only = e._reference_fk(c.base_pose(), c.reference)
        assert not np.allclose(arm_only, c.hand_goal, atol=1e-4)
        np.testing.assert_array_equal(obs["history"][-1, 20:25], 0)  # No IK joint target.
        np.testing.assert_array_equal(obs["history"][-1, 55:61], 0)  # No latent base solution.
        np.testing.assert_array_equal(obs["history"][-1, 77:87], 0)  # Masked target token.
        np.testing.assert_array_equal(obs["preview"][:, 1:7], 0)
    finally:
        e.close()
