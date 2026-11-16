from dataclasses import replace

import numpy as np
import pytest

from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.tracking import body_motion_cost


def config(**changes):
    return ManipulationConfig(
        robot="innate_mars",
        scenario="reach",
        action_mode="direct_velocity",
        trajectory_preview=True,
        tracking_reward=True,
        tracking_precision=4,
        balanced_grasp_setup=True,
        support_half_width=0.04,
        contact_cone="elliptic",
        contact_impratio=3,
        randomization_scale=0,
        hand_position_tolerance=0.005,
        hand_orientation_tolerance=0.05,
        **changes,
    )


@pytest.mark.parametrize("field", ["reach_hold_probability", "fixed_base_probability"])
@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
def test_hold_options_fail_closed(field, bad):
    with pytest.raises(ValueError):
        config(**{field: bad})


def test_quiet_body_cost_covers_vertical_and_roll_pitch_motion():
    assert body_motion_cost(np.zeros(6)) == 0
    for i in range(6):
        velocity = np.zeros(6)
        velocity[i] = 0.005 if i < 3 else 0.02
        assert body_motion_cost(velocity) == pytest.approx(0.5)
        assert 0.5 < body_motion_cost(velocity * 2) < 1
    with pytest.raises(ValueError):
        body_motion_cost(np.zeros(2))


@pytest.mark.sim
def test_pregrasp_posture_hold_is_reachable_fixed_and_requires_two_seconds():
    env = ManipulationEnv(config(reach_hold_probability=1, hold_pregrasp_fixture=True, horizon=3))
    try:
        obs, info = env.reset(seed=1861000)
        assert not info["initialization_failed"], info
        assert info["reach_hold_trial"] and env.control.phase == "pregrasp"
        assert env.arm_trajectory is None and env.retarget_goal is None
        np.testing.assert_allclose(env.control.hand_goal, env.control.ee_pose(), atol=1e-10)
        np.testing.assert_allclose(env.control.base_goal, env.control.base_pose(), atol=1e-10)
        assert env.control.base_mask.all()
        assert env.active_base_xy_tolerance == 0.005
        assert np.linalg.norm(env.data.xpos[env.payload] - env.control.ee_pose()[:3]) < 0.18
        for _ in range(25):
            obs, reward, done, truncated, info = env.step(np.zeros(8))
            assert not info["is_success"] and not done
        for _ in range(25):
            obs, reward, done, truncated, info = env.step(np.zeros(8))
        assert info["is_success"] and not done and not truncated
        assert info["ee_position_error"] < 0.005
        assert info["base_goal_error"] < 0.005
        assert not info["wheel_lift"] and not info["unsafe_contact"]
    finally:
        env.close()


@pytest.mark.sim
def test_fixed_base_pregrasp_keeps_current_base_without_removing_hand_motion():
    env = ManipulationEnv(replace(config(), scenario="pregrasp", fixed_base_probability=1, base_xy_tolerance=0.05))
    try:
        _, info = env.reset(seed=1861000)
        assert not info["initialization_failed"], info
        assert info["fixed_base_trial"] and not info["reach_hold_trial"]
        np.testing.assert_allclose(env.control.base_goal, env.control.base_pose(), atol=1e-10)
        assert env.control.base_mask.all()
        assert np.linalg.norm(env.control.hand_goal[:3] - env.control.ee_pose()[:3]) > 0.005
        assert env.active_base_xy_tolerance == 0.005
    finally:
        env.close()


@pytest.mark.sim
def test_pregrasp_hold_cannot_be_sampled_as_a_closed_empty_example():
    env = ManipulationEnv(
        config(reach_hold_probability=1, hold_pregrasp_fixture=True, empty_grasp_probability=1, retention_context=True)
    )
    try:
        _, info = env.reset(seed=1861000)
        assert info["reach_hold_trial"] and not info["initialization_failed"]
        assert not info["empty_grasp_trial"]
        assert env.control.grip_reference == env.profile.gripper_open
    finally:
        env.close()


@pytest.mark.sim
def test_local_pose_curriculum_preserves_fixed_base_and_certifies_orientation_witnesses():
    env = ManipulationEnv(
        config(
            reach_local_probability=1,
            hold_pregrasp_fixture=True,
            certify_reach_paths=True,
            dynamic_references=True,
            horizon=6,
            empty_grasp_probability=1,
            retention_context=True,
            wheel_feedback_gain=0.5,
            whole_body_settling=True,
        )
    )
    modes = set()
    try:
        for seed in range(64, 72):
            _, info = env.reset(seed=seed)
            assert not info["initialization_failed"], info
            assert info["reach_local_trial"] and not info["empty_grasp_trial"]
            assert not info["reach_hold_trial"] and env.control.base_mask.all()
            assert env.control.base_velocity_feedback_gain == 0.5
            assert env.arm_trajectory is not None and env.trajectory_admission["accepted"]
            assert env.retarget_goal is None and env.control.phase == "pregrasp"
            np.testing.assert_allclose(env.control.base_goal, env.control.base_pose(), atol=1e-10)
            modes.add(env.reach_local_mode)
            if env.reach_local_mode == "orientation":
                q = env.arm_trajectory.sample(env.arm_trajectory.end)[0]
                np.testing.assert_allclose(q[:-1], env.control.positions()[:-1], atol=1e-10)
                assert 0.08 <= abs(q[-1] - env.control.positions()[-1]) <= 0.20
        assert modes == {"orientation", "combined"}
    finally:
        env.close()


@pytest.mark.sim
@pytest.mark.parametrize("whole_body", [False, True])
def test_settling_cannot_reward_unmeasured_chassis_rocking(monkeypatch, whole_body):
    env = ManipulationEnv(
        config(reach_hold_probability=1, hold_pregrasp_fixture=True, whole_body_settling=whole_body, horizon=6)
    )
    try:
        env.reset(seed=1861000)
        env.settle_time = 1.99
        env.data.qvel[:] = 0
        env.data.qvel[env.control.base_dof + 4] = 0.04
        # Freeze the measured state to isolate the terminal predicate, rather
        # than depending on how much a physical solver damps this perturbation.
        monkeypatch.setattr(env, "_tick", lambda *args, **kwargs: {"force": 0, "bilateral_contact": False})
        _, _, _, _, info = env.step(np.zeros(8))
        assert info["is_success"] is (not whole_body)
    finally:
        env.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"wheel_feedback_gain": float("nan")},
        {"wheel_feedback_gain": -1},
        {"feedback_orientation_scale": 0},
        {"feedback_orientation_scale": 0.2},
        {"reach_local_probability": 0.3},
        {"reach_hold_probability": 0.8, "reach_local_probability": 0.4},
    ],
)
def test_new_curriculum_options_fail_closed(changes):
    with pytest.raises(ValueError):
        config(**changes)


@pytest.mark.sim
@pytest.mark.parametrize("kind", ["xyz", "yaw", "pitch"])
def test_fixed_base_training_uses_the_execution_envelope(monkeypatch, kind):
    import mujoco

    env = ManipulationEnv(
        config(reach_hold_probability=1, hold_pregrasp_fixture=True, fixed_base_contract=True, whole_body_settling=True)
    )
    try:
        _, initial = env.reset(seed=1861000)
        assert initial["base_heading_tolerance"] == 0.01
        assert initial["fixed_base_contract_active"]

        def shifted(*args, **kwargs):
            if kind == "xyz":
                env.data.qpos[0] += 0.012
            else:
                quat = np.zeros(4)
                mujoco.mju_axisAngle2Quat(
                    quat, np.array([0.0, 0.0, 1.0]) if kind == "yaw" else np.array([0.0, 1.0, 0.0]), 0.03
                )
                env.data.qpos[3:7] = quat
            env.data.qvel[:] = 0
            mujoco.mj_forward(env.model, env.data)
            return {"force": 0.0, "bilateral_contact": False}

        monkeypatch.setattr(env, "_tick", shifted)
        _, _, done, truncated, info = env.step(np.zeros(8))
        assert done and not truncated and env.control.stopped
        assert info["code"] == "fixed_base_contract_violation"
        assert info["fixed_base_contract_violation"]
        assert info["reward_terms"]["failure_penalty"] == 20
    finally:
        env.close()


@pytest.mark.sim
def test_goal_update_without_motion_cannot_manufacture_progress(monkeypatch):
    env = ManipulationEnv(config(reach_hold_probability=1, hold_pregrasp_fixture=True, pose_progress_weight=1))
    try:
        env.reset(seed=1861000)

        def changed_goal(*args, **kwargs):
            hand = env.control.hand_goal.copy()
            hand[0] += 0.1
            env.control.set_pose_targets(env.control.base_goal, env.control.base_mask, hand, True)
            return {"force": 0.0, "bilateral_contact": False}

        monkeypatch.setattr(env, "_tick", changed_goal)
        _, _, _, _, info = env.step(np.zeros(8))
        assert info["reward_terms"]["pose_progress_reward"] == pytest.approx(0, abs=1e-12)
    finally:
        env.close()


def test_progress_signal_distinguishes_fine_errors_and_full_body_summary_rejects_rocking():
    from emet.rl.manipulation.env import pose_progress_potential
    from emet.rl.manipulation.evaluation import base_is_still

    errors = [pose_progress_potential(p, 0) for p in [0.005, 0.010, 0.020]]
    assert errors[0] - errors[1] > 0.1 and errors[1] - errors[2] > 0.1
    assert base_is_still({"base_linear_velocity_m_s": 0, "base_yaw_velocity_rad_s": 0})
    assert not base_is_still({"base_linear_norm_m_s": 0, "base_angular_norm_rad_s": 0.04})


@pytest.mark.sim
@pytest.mark.parametrize("hand_active", [False, True])
def test_progress_only_rewards_motion_toward_an_active_hand(monkeypatch, hand_active):
    import mujoco

    env = ManipulationEnv(config(reach_hold_probability=1, hold_pregrasp_fixture=True, pose_progress_weight=1))
    try:
        env.reset(seed=1861000)
        hand = env.control.hand_goal.copy()
        hand[0] += 0.02
        env.control.set_pose_targets(env.control.base_goal, env.control.base_mask, hand, hand_active)

        def moved(*args, **kwargs):
            env.data.qpos[0] += 0.001
            env.data.qvel[:] = 0
            mujoco.mj_forward(env.model, env.data)
            return {"force": 0.0, "bilateral_contact": False}

        monkeypatch.setattr(env, "_tick", moved)
        _, _, _, _, info = env.step(np.zeros(8))
        progress = info["reward_terms"]["pose_progress_reward"]
        assert progress > 0 if hand_active else progress == 0
    finally:
        env.close()
