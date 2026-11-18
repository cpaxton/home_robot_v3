"""Whole-body witnesses obey wheel kinematics and bounded joint derivatives."""

import numpy as np
import pytest

from emet.rl.manipulation.whole_body_splines import JointWheelPath, WholeBodySplineTrackingEnv, manifest


def path(arm=None, wheels=None, **kwargs):
    return JointWheelPath(
        np.zeros((3, 2)) if arm is None else arm,
        [[0, 0], [0.1, 0.1], [0.2, 0.2]] if wheels is None else wheels,
        kwargs.pop("base", [0.1, -0.2, 0.7]),
        kwargs.pop("axle_offset", [0.04, -0.01]),
        kwargs.pop("separation", 0.4),
        [0.2, 0.6, 0.6, 0.6],
        [0.2, 0.6, 1.2, 1.2],
        [0.8, 2.4, 4.8, 4.8],
        **kwargs,
    )


def test_straight_motion_preserves_heading_and_measured_start():
    p = path()
    start, _ = p.coordinates(0)
    end, _ = p.coordinates(p.end)
    np.testing.assert_allclose(start, [0.1, -0.2, 0.7], atol=1e-12)
    np.testing.assert_allclose(end, start + [0.2 * np.cos(0.7), 0.2 * np.sin(0.7), 0], atol=1e-10)
    np.testing.assert_array_equal(p.derivatives(0.1), np.zeros((3, 4)))


def test_turn_rotates_base_about_axle_and_wraps_heading():
    p = path(wheels=[[0, 0], [-0.02, 0.02], [-0.04, 0.04]], base=[0, 0, np.pi - 0.1])
    expected_axle = p.start_base[:2] + p.rotation(p.start_base[2]) @ p.offset
    for phase in np.linspace(0, 1, 21):
        base = p.base_at(phase)
        axle = base[:2] + p.rotation(base[2]) @ p.offset
        np.testing.assert_allclose(axle, expected_axle, atol=1e-11)
    assert p.base_at(1)[2] == pytest.approx(-np.pi + 0.1)


def test_reversing_curved_wheel_path_has_no_lateral_axle_motion():
    p = path(
        arm=[[0, 0], [0.15, -0.10], [-0.05, 0.15], [0, 0]],
        wheels=[[0, 0], [0.08, 0.10], [0.03, 0.15], [-0.02, 0.04]],
    )
    for phase in np.linspace(0.01, 0.99, 101):
        delta = (p.axle_xy(phase + 1e-6) - p.axle_xy(phase - 1e-6)) / 2e-6
        yaw = p.start_base[2] + p.geometry(phase)[1]
        assert abs(delta @ [-np.sin(yaw), np.cos(yaw)]) < 1e-8
    for time in np.linspace(p.initial_hold_s, p.end, 1001):
        observed = abs(p.derivatives(time))
        assert np.all(observed <= p.bounds * (1 + 1e-8))
        assert np.all(observed <= np.array([p.speed, p.acceleration, p.jerk]) * (1 + 1e-8))
    for time in (p.initial_hold_s, p.end):
        np.testing.assert_allclose(p.derivatives(time)[:2], 0, atol=1e-10)


def test_manifest_has_independent_splits_postures_and_speed_bands():
    spec = manifest()
    assert spec == manifest()
    assert len({c["id"] for c in spec["cases"]}) == len(spec["cases"])
    assert len({c["seed"] for c in spec["cases"]}) == len(spec["cases"])
    assert {c["posture"] for c in spec["cases"]} == {"ready", "raised", "left_reach", "folded_right", "joint_space"}
    assert [x["speed_fraction"] for x in spec["levels"]] == [0.25, 0.55, 0.85]
    for c in spec["cases"]:
        other = "train" if c["split"] == "audit" else "audit"
        assert c["seed"] not in {x["seed"] for x in spec["cases"] if x["split"] == other}


@pytest.mark.parametrize("change", [{"separation": True}, {"initial_hold_s": -1}, {"wheels": [[1, 0]] * 3}])
def test_invalid_kinematics_fail_closed(change):
    with pytest.raises(ValueError):
        path(**change)


def test_invalid_case_rejected_before_constructing_physics():
    case = manifest()["cases"][0]
    with pytest.raises(ValueError, match="identity"):
        WholeBodySplineTrackingEnv([{**case, "split": "audit"}])


@pytest.mark.sim
def test_fk_reference_changes_base_and_arm_without_runtime_state_writes():
    case = manifest()["cases"][0]
    env = WholeBodySplineTrackingEnv([case])
    try:
        observation, info = env.reset()
        assert env.observation_space.contains(observation)
        e = env.unwrapped
        before = {k: getattr(e.data, k).copy() for k in ("qpos", "qvel", "ctrl")}
        now = float(e.data.time)
        for time in np.linspace(0, env.reference.end, 21):
            env.reference.sample(time)
        for key, value in before.items():
            np.testing.assert_array_equal(getattr(e.data, key), value)
        assert float(e.data.time) == now
        assert info["trajectory_spread"]["base"]["path_length_m"] > 0.01
        assert max(info["trajectory_spread"]["joint_span_rad"]) > 0.1
        assert e.control.base_xy_tolerance == e.active_base_xy_tolerance == 0.05
    finally:
        env.close()
