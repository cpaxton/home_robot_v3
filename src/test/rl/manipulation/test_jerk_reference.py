"""Reference geometry, derivative limits and retiming are independent gates."""

import numpy as np
import pytest

from emet.rl.manipulation.jerk_reference import SmoothPhase, compile_joint_path, polynomial_peaks, simplify_collinear


def test_collinear_densification_is_removed_without_cutting_corners_or_reversals():
    line = np.linspace([0, 0], [0.3, -0.1], 41)
    np.testing.assert_allclose(simplify_collinear(line), line[[0, -1]])
    corner = np.vstack([line, [0.3, 0.1], [0.3, 0.1], [0.2, 0.1], [0.3, 0.1]])
    np.testing.assert_allclose(simplify_collinear(corner), [[0, 0], [0.3, -0.1], [0.3, 0.1], [0.2, 0.1], [0.3, 0.1]])


def test_analytic_bounds_cover_random_filtered_lag_retiming_and_real_corner_stops():
    speed, acceleration, jerk = np.array([0.6, 0.4]), np.array([1.2, 0.8]), np.array([4.8, 3.2])
    path, certificate = compile_joint_path(
        [[0, 0], [0.1, -0.04], [0.2, -0.08], [0.1, 0.03]],
        speed,
        acceleration,
        jerk,
        phase_acceleration=2,
        phase_jerk=8,
    )
    assert certificate["retained_vertices"] == 3
    for i, part in enumerate(path.segments):
        v, a, j = polynomial_peaks(part)
        for value, limit in zip((v, a + 2 * v, j + 6 * a + 8 * v), (speed, acceleration, jerk), strict=True):
            assert np.all(value <= limit * (1 + 1e-9))
        if i:
            for before, after in zip(
                path.segments[i - 1].sample(part.start)[:3], part.sample(part.start)[:3], strict=True
            ):
                np.testing.assert_allclose(before, after, atol=1e-9)
    phase = SmoothPhase(path.end)
    rng = np.random.default_rng(123)
    for k in range(15000):
        request = float(rng.uniform()) if k % 33 == 0 else phase.request
        before = phase.acceleration
        phase.advance(request, 0)
        assert phase.acceleration == before  # A rate-request step cannot jump acceleration.
        phase.advance(request, 0.002)
        assert 0 <= phase.rate <= 1 and abs(phase.acceleration) <= 2 and abs(phase.jerk) <= 8
        _, v, a, j = path.sample(phase.progress)
        actual = (
            v * phase.rate,
            a * phase.rate**2 + v * phase.acceleration,
            j * phase.rate**3 + 3 * a * phase.rate * phase.acceleration + v * phase.jerk,
        )
        for value, limit in zip(actual, (speed, acceleration, jerk), strict=True):
            assert np.all(abs(value) <= limit * (1 + 1e-8))
        if phase.progress >= path.end:
            break
    else:
        pytest.fail("filtered phase failed to finish the finite path")


def test_phase_integration_prediction_and_past_preview_are_consistent():
    phase = SmoothPhase(10)
    phase.advance(1, 0.2)
    expected = phase.predict(0.05)
    prior = phase.progress
    phase.advance(1, 0.05)
    np.testing.assert_allclose([phase.progress, phase.first, phase.rate], expected, atol=1e-12)
    assert phase.preview_progress(-0.05) == pytest.approx(prior)
    assert phase.preview_progress(0) == phase.progress
    assert phase.preview_progress(1) >= phase.progress
    h = 1e-5
    assert (phase.predict(h)[0] - phase.progress) / h == pytest.approx(phase.rate, abs=1e-5)


@pytest.mark.parametrize(
    "change", [{"speed": [0]}, {"acceleration": [float("nan")]}, {"jerk": [-1]}, {"phase_jerk": -1}]
)
def test_invalid_reference_limits_fail_before_a_curve_is_returned(change):
    args = {"path": [[0], [0.1]], "speed": [1], "acceleration": [1], "jerk": [1]}
    args.update(change)
    with pytest.raises(ValueError):
        compile_joint_path(**args)


@pytest.mark.sim
@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
def test_continuous_fk_twist_and_acceleration_match_finite_differences_without_live_edits(robot):
    from scipy.spatial.transform import Rotation

    from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
    from emet.rl.manipulation.jerk_pose_reference import FKPoseReference
    from emet.rl.manipulation.tracking import TaskFrameFK

    env = ManipulationEnv(ManipulationConfig(robot=robot, scenario="reach", randomization_scale=0))
    try:
        env.reset(seed=200002)
        c, profile = env.control, env.profile
        q = c.positions().copy()
        goal = np.clip(q + np.linspace(0.02, -0.01, profile.arm_size), c.limits[:, 0], c.limits[:, 1])
        path, cert = compile_joint_path(
            [q, goal], profile.speed, profile.acceleration, 4 * np.asarray(profile.acceleration)
        )
        before = [env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), float(env.data.time)]
        ref = FKPoseReference(TaskFrameFK(env.model, env.data, c), c.base_pose(), path, c.grip_reference, cert)
        t, h = path.end * 0.43, 1e-4
        sample = ref.sample_motion(t)
        left, right = ref.sample(t - h)[1], ref.sample(t + h)[1]

        def rotation(p):
            cols = p[3:].reshape(3, 2)
            return np.column_stack([cols, np.cross(cols[:, 0], cols[:, 1])])

        velocity = np.r_[
            (right[:3] - left[:3]) / (2 * h),
            Rotation.from_matrix(rotation(right) @ rotation(left).T).as_rotvec() / (2 * h),
        ]
        np.testing.assert_allclose(sample["hand_twist_world"], velocity, atol=1e-7, rtol=1e-4)
        acceleration = (
            np.array(ref.sample_motion(t + h)["hand_twist_world"]) - ref.sample_motion(t - h)["hand_twist_world"]
        ) / (2 * h)
        np.testing.assert_allclose(sample["hand_acceleration_world"], acceleration, atol=1e-6, rtol=1e-3)
        for actual, expected in zip(
            [env.data.qpos, env.data.qvel, env.data.ctrl, float(env.data.time)], before, strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
        assert ref.record()["format"] == "task_pose_fk_reference_v2"
        np.testing.assert_allclose(ref.sample(ref.end)[1], ref.sample(ref.end + 1)[1], atol=1e-12)
        assert not np.any(ref.sample_motion(ref.end + 1)["hand_twist_world"])
    finally:
        env.close()


@pytest.mark.sim
def test_original_cartesian_endpoint_and_c2_terminal_correction_are_preserved_and_not_sent_as_v1():
    from types import SimpleNamespace

    from scipy.spatial.transform import Rotation

    from emet.rl.manipulation.command_runtime import send_task_pose_trajectory
    from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
    from emet.rl.manipulation.jerk_pose_reference import FKPoseReference
    from emet.rl.manipulation.tracking import TaskFrameFK

    env = ManipulationEnv(ManipulationConfig(robot="innate_mars", scenario="reach", randomization_scale=0))
    try:
        env.reset(seed=200002)
        c = env.control
        q = c.positions().copy()
        path, cert = compile_joint_path(
            [q, q + [0, 0.01, -0.01, 0, 0]],
            env.profile.speed,
            env.profile.acceleration,
            4 * np.array(env.profile.acceleration),
        )
        fk = TaskFrameFK(env.model, env.data, c)
        goal = fk(c.base_pose(), path.sample(path.end)[0]).copy()
        cols = goal[3:].reshape(3, 2)
        rot = np.column_stack([cols, np.cross(cols[:, 0], cols[:, 1])])
        goal[:3] += [0.001, -0.0005, 0]
        goal[3:] = (Rotation.from_rotvec([0, 0, 0.002]).as_matrix() @ rot)[:, :2].ravel()
        ref = FKPoseReference(fk, c.base_pose(), path, c.grip_reference, cert, goal_pose=goal)
        assert ref.end > path.end
        np.testing.assert_allclose(ref.sample(ref.end)[1], goal, atol=1e-12)
        for t in [path.end - 1e-7, path.end, path.end + 1e-7, ref.end]:
            motion = ref.sample_motion(t)
            assert np.linalg.norm(motion["hand_twist_world"]) < 1e-7
            assert np.linalg.norm(motion["hand_acceleration_world"]) < 1e-4
        with pytest.raises(ValueError, match="unsupported task-pose"):
            send_task_pose_trajectory(SimpleNamespace(_state={"task_pose_control": True}), ref)
        assert not ref.record()["terminal_correction"]["joint_witness_certified"]
    finally:
        env.close()
