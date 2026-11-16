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
