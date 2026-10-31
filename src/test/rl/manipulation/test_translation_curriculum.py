import numpy as np
import pytest

from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.translation_curriculum import sample_translation_reference


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
def test_translation_probability_fails_closed(bad):
    with pytest.raises(ValueError):
        ManipulationConfig(reach_local_translation_probability=bad)


def test_translation_sampling_requires_local_reach_curriculum():
    with pytest.raises(ValueError, match="local reach"):
        ManipulationConfig(reach_local_translation_probability=0.5)


@pytest.mark.sim
@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
def test_translation_witnesses_preserve_attitude_and_do_not_modify_live_state(robot):
    env = ManipulationEnv(
        ManipulationConfig(
            robot=robot,
            scenario="reach",
            randomization_scale=0,
            reach_hold_probability=1 if robot == "innate_mars" else 0,
            hold_pregrasp_fixture=robot == "innate_mars",
            balanced_grasp_setup=robot == "innate_mars",
            support_half_width=0.04,
        )
    )
    try:
        env.reset(seed=1861000)
        before = [np.asarray(x).copy() for x in (env.data.qpos, env.data.qvel, env.data.ctrl)]
        directions = []
        for seed in range(8):
            q, record = sample_translation_reference(env.control, np.random.default_rng(seed))
            assert 0.005 <= record["distance_m"] <= 0.020
            assert record["orientation_change_rad"] <= 0.002
            assert np.all(q >= env.control.limits[:, 0] + 0.02)
            assert np.all(q <= env.control.limits[:, 1] - 0.02)
            actual = env._fk_hand_pose(env.control.base_pose(), q)
            np.testing.assert_allclose(actual, record["goal_hand"], atol=1e-8)
            directions.append(record["translation_world_m"])
        assert np.linalg.matrix_rank(directions, tol=1e-5) >= 2
        for actual, expected in zip((env.data.qpos, env.data.qvel, env.data.ctrl), before, strict=True):
            np.testing.assert_array_equal(actual, expected)
    finally:
        env.close()


@pytest.mark.sim
def test_translation_reach_goals_have_separate_path_admission_and_no_empty_grasp_label():
    env = ManipulationEnv(
        ManipulationConfig(
            robot="innate_mars",
            scenario="reach",
            randomization_scale=0,
            reach_local_probability=1,
            reach_local_translation_probability=1,
            action_mode="direct_velocity",
            trajectory_preview=True,
            tracking_reward=True,
            hold_pregrasp_fixture=True,
            balanced_grasp_setup=True,
            contact_cone="elliptic",
            contact_impratio=3,
            support_half_width=0.04,
            certify_reach_paths=True,
            dynamic_references=True,
            horizon=6,
            retention_context=True,
            empty_grasp_probability=1,
        )
    )
    try:
        for seed in range(64, 68):
            obs, info = env.reset(seed=seed)
            assert not info["initialization_failed"]
            assert info["reach_local_mode"] == "translation" and not info["empty_grasp_trial"]
            assert env.trajectory_admission["accepted"]
            assert env.arm_trajectory is not None and env.control.base_mask.all()
            assert set(obs) == {"history", "preview"}
            final = env.arm_trajectory.sample(env.arm_trajectory.end)[0]
            np.testing.assert_allclose(
                env._fk_hand_pose(env.control.base_goal, final), info["reach_local_translation"]["goal_hand"], atol=1e-8
            )
    finally:
        env.close()
