"""Reachable arm targets and policy scope must match training and execution."""

import mujoco
import numpy as np
import pytest
import torch

from emet.rl.arm import ArmHistoryFeatures
from emet.rl.control import baseline_action
from emet.rl.env import EnvConfig, MarsReachEnv


def test_arm_configuration_rejects_base_task():
    with pytest.raises(ValueError, match="arm_only"):
        EnvConfig(task="base", arm_only=True)
    with pytest.raises(ValueError, match="stage 0"):
        EnvConfig(arm_only=True, stage=2)


@pytest.mark.parametrize("seed", [1, 17, 93, 70000])
def test_arm_sampler_reachable_path_and_repeatable(seed):
    env = MarsReachEnv(EnvConfig(arm_only=True))
    try:
        obs, _ = env.reset(seed=seed)
        start = env.data.qpos.copy()
        goal = env.control.goal.arm.copy()
        pose = env.control.goal_pose.copy()
        assert env.control.goal.mask[0] == 0
        assert np.all(goal >= env.control.limits[:, 0] + 0.08)
        assert np.all(goal <= env.control.limits[:, 1] - 0.08)
        count = max(2, int(np.ceil(np.max(abs(goal - start[env.control.arm_q])) / 0.04)) + 1)
        for t in np.linspace(0, 1, count):
            env.data.qpos[env.control.arm_q] = start[env.control.arm_q] + t * (goal - start[env.control.arm_q])
            mujoco.mj_forward(env.model, env.data)
            assert env.control.obstacle_features()[1] > 0.0005
        np.testing.assert_allclose(env.control.pose_features(), pose, atol=1e-10)
        obs2, _ = env.reset(seed=seed)
        for key in obs:
            np.testing.assert_array_equal(obs[key], obs2[key])
        np.testing.assert_array_equal(env.control.goal.arm, goal)
    finally:
        env.close()


def test_arm_feedback_prior_matches_teacher_after_motion():
    env = MarsReachEnv(EnvConfig(arm_only=True))
    try:
        obs, _ = env.reset(seed=19)
        features = ArmHistoryFeatures(env.observation_space)
        for _ in range(5):
            prior = features({k: torch.from_numpy(v[None]) for k, v in obs.items()})[0, -7:].detach().numpy()
            np.testing.assert_allclose(prior[2:], baseline_action(env.control)[2:], atol=2e-6)
            obs, _, done, truncated, _ = env.step(np.ones(7) * 0.1)
            np.testing.assert_array_equal(env.control.last_action[:2], 0)
            if done or truncated:
                break
    finally:
        env.close()


def test_arm_policy_refuses_base_goals():
    from emet.rl.policy import PPOPolicy

    policy = PPOPolicy.__new__(PPOPolicy)
    policy.manifest = {"training": {"env": {"arm_only": True}}}
    with pytest.raises(ValueError, match="cannot execute a base goal"):
        policy({"goal": np.ones(15)})


def test_arm_waypoints_require_final_measured_arrival():
    from emet.rl.trajectories import evaluate_trajectories

    report = evaluate_trajectories(arm=True, episodes=2, seed=123000)
    assert report["success_rate"] == 1
    for row in report["episodes"]:
        assert row["last_waypoint"] == row["waypoints"] - 1
        assert row["joint_error"] < 0.05
        assert row["collision"] is False


def test_coordinated_pose_trajectory_uses_reachable_base_and_hand_waypoints():
    from emet.rl.trajectories import evaluate_trajectories

    report = evaluate_trajectories(coordinated=True, episodes=1, seed=120000)
    assert report["coordinated"] is True
    assert report["success_rate"] == 1
    row = report["episodes"][0]
    assert row["waypoints"] == 5
    assert row["last_waypoint"] == 4
    assert row["collision"] is False
    assert row["xy_error"] <= 0.05
    assert row["yaw_error"] <= np.deg2rad(5)
    assert row["ee_position_error"] <= 0.03
    assert row["ee_orientation_error"] <= 0.12


def test_training_environment_advances_measured_coordinated_waypoints():
    env = MarsReachEnv(
        EnvConfig(task="coordinated", pose_targets=True, trajectory_waypoints=5, horizon_seconds=15)
    )
    try:
        env.reset(seed=120000)
        advanced = 0
        for _ in range(300):
            _, _, terminated, truncated, info = env.step(env.expert_action())
            advanced += info["waypoint_advanced"]
            if terminated or truncated:
                break
        assert advanced == 4
        assert info["is_success"] is True
        assert info["trajectory_waypoint"] == 4
        assert info["trajectory_waypoints"] == 5
    finally:
        env.close()


def test_coordinated_waypoint_distribution_uses_turn_drive_align_instead_of_strafing():
    from emet.rl.control import Goal
    from emet.rl.waypoints import coordinated_waypoints

    start = np.zeros(3)
    final = Goal(np.array([0, 0.06, 0]), np.full(5, 0.1), np.ones(6))
    goals = coordinated_waypoints(start, np.zeros(5), final, 5)
    assert len(goals) == 5
    previous = start
    for goal in goals:
        delta = goal.base[:2] - previous[:2]
        if np.linalg.norm(delta) > 1e-6:
            heading = np.array([np.cos(previous[2]), np.sin(previous[2])])
            assert abs(delta @ heading) / np.linalg.norm(delta) > 0.999
        previous = goal.base
    np.testing.assert_allclose(goals[-1].base, final.base)
    np.testing.assert_allclose(goals[-1].arm, final.arm)
