"""Executable base-reference geometry, derivative bounds and serialization scope."""

from types import SimpleNamespace

import numpy as np
import pytest

from emet.rl.manipulation.base_goal_reference import BaseGoalReference
from emet.rl.manipulation.jerk_reference import polynomial_peaks
from emet.rl.manipulation.profiles import PROFILES


@pytest.fixture
def control(monkeypatch):
    class FK:
        def __init__(self, *args):
            pass

        def __call__(self, base, joints):
            c, s = np.cos(base[2]), np.sin(base[2])
            return np.r_[base[:2], 0.7, c, -s, s, c, 0, 0]

    monkeypatch.setattr("emet.rl.manipulation.base_goal_reference.TaskFrameFK", FK)
    return SimpleNamespace(
        profile=PROFILES["innate_mars"], model=None, data=None,
        base_pose=lambda: np.array([0.1, -0.1, 0.2]), axle_offset=np.array([0.05, -0.01]),
        positions=lambda: np.zeros(5), grip_reference=0.3,
    )


@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
@pytest.mark.parametrize("delta", [[0, 0, 0], [0, 0.15, 0], [0, -0.15, 0], [-0.15, 0, -0.2], [0.1, 0.1, 0.4]])
def test_base_goal_reference_reaches_endpoint_with_c2_axle_bounds(control, robot, delta):
    control.profile = PROFILES[robot]
    start = control.base_pose()
    target = start + delta
    path = BaseGoalReference(control, target)
    np.testing.assert_allclose(path.sample(-1)[0], start, atol=1e-10)
    np.testing.assert_allclose(path.sample(path.end + 1)[0], target, atol=1e-9)
    assert not path.hand_mask and path.base_mask.all()
    assert not path.certificate["dynamic_feasibility_verified"]
    assert not path.certificate["initial_motion_boundary_verified"]
    for time, kind, scalar, _, _ in path.parts:
        p = control.profile
        speed, acceleration = (p.base_speed, p.base_acceleration) if kind == "drive" else (p.base_yaw_speed, p.base_yaw_acceleration)
        for part in scalar.segments:
            v, a, j = polynomial_peaks(part)
            assert v[0] <= speed * (1 + 1e-8)
            assert a[0] <= acceleration * (1 + 1e-8)
            assert j[0] <= 4 * acceleration * (1 + 1e-8)
        for endpoint in (0, scalar.end):
            _, v, a, _ = scalar.sample(endpoint)
            np.testing.assert_allclose(np.r_[v, a], 0, atol=1e-9)
        if time:
            np.testing.assert_allclose(path.coordinates(time - 1e-7)[0], path.coordinates(time + 1e-7)[0], atol=1e-8)
    record = path.record()
    assert record["format"] != "task_pose_trajectory_v1"
    assert record["format"] == "emet_base_goal_reference_v1"


def test_drive_axle_has_no_lateral_velocity_even_with_offset_base_link(control):
    path = BaseGoalReference(control, [0.1, 0.1, -0.1])
    for time in np.linspace(0.01, path.end - 0.01, 200):
        before, _ = path.coordinates(time - 1e-5)
        after, _ = path.coordinates(time + 1e-5)
        middle, _ = path.coordinates(time)
        def axle(base):
            c, s = np.cos(base[2]), np.sin(base[2])
            return base[:2] + np.array([[c, -s], [s, c]]) @ control.axle_offset[:2]
        velocity = (axle(after) - axle(before)) / 2e-5
        lateral = np.array([-np.sin(middle[2]), np.cos(middle[2])]) @ velocity
        assert abs(lateral) < 1e-8


def test_final_heading_wrap_does_not_compile_a_full_spin(control):
    control.base_pose = lambda: np.array([0.0, 0.0, np.pi - 0.01])
    control.axle_offset[:] = 0
    path = BaseGoalReference(control, [0, 0, -np.pi + 0.01])
    assert len(path.parts) == 1
    assert path.end < 1
    assert path.parts[0][2].sample(path.end)[0][0] == pytest.approx(0.02)


@pytest.mark.parametrize("target", [[0, 0], [0, 0, np.nan], [0, np.inf, 0]])
def test_malformed_goal_rejected_without_mutating_control(control, target):
    start = control.base_pose().copy()
    with pytest.raises(ValueError):
        BaseGoalReference(control, target)
    np.testing.assert_array_equal(control.base_pose(), start)


@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
@pytest.mark.parametrize("delta", [[0.16, 0, 0], [-0.16, 0, 0], [0, 0.16, 0], [0, 0, -0.4]])
def test_replans_match_announced_pose_velocity_and_acceleration_through_all_phases(control, robot, delta):
    control.profile = PROFILES[robot]
    start = control.base_pose()
    previous = BaseGoalReference(control, start + delta)
    target = start + np.array([-0.07, 0.05, 0.1])
    for stamp in np.linspace(0.001, previous.end - 0.001, 31):
        path = BaseGoalReference(control, target, previous=previous, elapsed=stamp)
        before, after = previous.state(stamp), path.state(0)
        for index in (0, 1, 2, 3):
            np.testing.assert_allclose(after[index], before[index], atol=1e-9)
        for offset in (-0.001, -0.05):
            np.testing.assert_allclose(path.sample(offset)[0], previous.sample(stamp + offset)[0], atol=1e-9)
        np.testing.assert_allclose(path.sample(path.end)[0], target, atol=1e-9)
        assert path.boundary["announced_motion_boundary_matched"]
        assert not path.boundary["measured_motion_boundary_verified"]
        assert path.record()["format"] == "emet_base_goal_reference_v2"
        for offset, kind, scalar, _, _ in path.parts:
            p = control.profile
            speed, acceleration = (p.base_speed, p.base_acceleration) if kind == "drive" else (p.base_yaw_speed, p.base_yaw_acceleration)
            for segment in scalar.segments:
                v, a, j = polynomial_peaks(segment)
                assert v[0] <= speed + 1e-9
                assert a[0] <= acceleration + 1e-9
                assert j[0] <= 4 * acceleration + 1e-9
            if offset:
                left, right = path.state(offset - 1e-7), path.state(offset + 1e-7)
                for index in (0, 2, 3):
                    np.testing.assert_allclose(left[index], right[index], atol=1e-6)
    np.testing.assert_array_equal(control.base_pose(), start)


def test_repeated_replans_during_braking_and_after_completion_preserve_boundaries(control):
    previous = BaseGoalReference(control, control.base_pose() + [0.2, 0, 0])
    stamp = 0.5
    for i in range(12):
        target = control.base_pose() + [0.1, (-1)**i * 0.05, 0.1]
        path = BaseGoalReference(control, target, previous=previous, elapsed=stamp)
        for index in (0, 2, 3):
            np.testing.assert_allclose(path.state(0)[index], previous.state(stamp)[index], atol=1e-9)
        np.testing.assert_allclose(path.sample(-0.001)[0], previous.sample(stamp - 0.001)[0], atol=1e-9)
        assert len(path._past_parts) <= 3
        previous, stamp = path, 0.05
    path = BaseGoalReference(control, control.base_pose(), previous=previous, elapsed=previous.end + 1)
    np.testing.assert_allclose(path.state(0)[2:4], 0, atol=1e-9)


def test_replan_rejects_incompatible_owner_clock_and_phase_bounds(control):
    previous = BaseGoalReference(control, control.base_pose() + [0.1, 0, 0])
    for kwargs in ({"elapsed": -1}, {"elapsed": np.nan}, {"elapsed": 0.1, "phase_acceleration": 1}):
        with pytest.raises(ValueError):
            BaseGoalReference(control, control.base_pose(), previous=previous, **kwargs)
    other = SimpleNamespace(**vars(control))
    with pytest.raises(ValueError, match="same unretimed"):
        BaseGoalReference(other, control.base_pose(), previous=previous, elapsed=0.1)


@pytest.mark.parametrize("v,a", [(0.12, 0.12), (0.001, -0.12), (0, 0.1), (np.nan, 0), (0.2, 0)])
def test_impossible_braking_boundaries_fail_closed(v, a):
    from emet.rl.manipulation.base_braking import braking_path

    with pytest.raises(ValueError):
        braking_path(v, a, 0.12, 0.12, 0.48)


def test_replan_central_difference_twist_is_continuous_at_installation(control):
    previous = BaseGoalReference(control, control.base_pose() + [0.15, 0, 0])
    stamp, epsilon = 0.7, 0.001
    path = BaseGoalReference(control, control.base_pose() + [0, 0.1, 0], previous=previous, elapsed=stamp)
    before = (previous.sample(stamp + epsilon)[0] - previous.sample(stamp - epsilon)[0]) / (2 * epsilon)
    after = (path.sample(epsilon)[0] - path.sample(-epsilon)[0]) / (2 * epsilon)
    np.testing.assert_allclose(after, before, atol=1e-6)
