"""Motion magnitude, cadence, masks, closed loops and bounded timing."""

import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from emet.rl.manipulation.reference_timing import retime_derivative_bounds
from emet.rl.manipulation.trajectory_metrics import SpreadRequirements, trajectory_spread


def hand(position, rotation=None):
    n = len(position)
    if rotation is None:
        rotation = np.tile(np.eye(3), (n, 1, 1))
    return np.column_stack([position, rotation[:, :, :2].reshape(n, 6)])


def test_base_motion_cannot_masquerade_as_arm_spread():
    times = np.linspace(0, 2, 101)
    yaw = np.linspace(0, 0.3, len(times))
    rb = Rotation.from_euler("z", yaw).as_matrix()
    base = np.column_stack([0.1 * times, np.zeros(len(times)), yaw])
    position = np.einsum("nij,j->ni", rb, [0.3, 0, 0.2]) + np.column_stack([base[:, :2], np.zeros(len(base))])
    result = trajectory_spread(times, hand(position, rb), base)
    assert result["world_hand"]["maximum_excursion_m"] > 0.1
    assert result["base_relative_hand"]["maximum_excursion_m"] < 1e-12
    assert result["base_relative_hand"]["rotation_excursion_rad"] < 1e-12
    assert not SpreadRequirements().accepts(result)


def test_return_to_start_still_has_nonzero_excursion_and_path():
    times = np.array([0, 1, 2.0])
    position = np.array([[0, 0, 0.2], [0.1, 0, 0.2], [0, 0, 0.2]])
    result = trajectory_spread(times, hand(position), np.zeros((3, 3)))
    assert result["world_hand"]["endpoint_displacement_m"] == 0
    assert result["world_hand"]["maximum_excursion_m"] == pytest.approx(0.1)
    assert result["world_hand"]["path_length_m"] == pytest.approx(0.2)
    assert SpreadRequirements().accepts(result)


def test_duty_is_time_weighted_and_inactive_hand_is_ignored():
    times = np.array([0.0, 1, 10])
    position = np.array([[0, 0, 0.2], [0.1, 0, 0.2], [0.1, 0, 0.2]])
    result = trajectory_spread(times, hand(position), np.zeros((3, 3)))
    assert result["reference_motion_fraction"] == pytest.approx(0.1)
    masked = trajectory_spread(times, hand(position), np.zeros((3, 3)), hand_mask=False)
    assert masked["reference_motion_fraction"] == 0 and masked["world_hand"] is None
    assert not SpreadRequirements().accepts(masked)


def test_orientation_and_base_yaw_cross_pi_by_short_path():
    angles = [np.pi - 0.01, -np.pi + 0.01]
    r = Rotation.from_euler("z", angles).as_matrix()
    result = trajectory_spread([0, 1], hand(np.zeros((2, 3)), r), np.column_stack([np.zeros((2, 2)), angles]))
    assert result["world_hand"]["rotation_travel_rad"] == pytest.approx(0.02)
    assert result["base"]["yaw_travel_rad"] == pytest.approx(0.02)
    assert result["base_relative_hand"]["rotation_travel_rad"] < 1e-12


@pytest.mark.parametrize("kind", ["times", "nan", "rotation", "mask"])
def test_bad_metrics_fail_closed(kind):
    t = np.array([0.0, 1])
    h = hand(np.zeros((2, 3)))
    b = np.zeros((2, 3))
    kwargs = {}
    if kind == "times":
        t[:] = 0
    if kind == "nan":
        h[0, 0] = np.nan
    if kind == "rotation":
        h[0, 3] = 2
    if kind == "mask":
        kwargs["base_mask"] = [1, 1, 1]
    with pytest.raises(ValueError):
        trajectory_spread(t, h, b, **kwargs)


def test_tiny_joint_path_loses_artificial_four_second_floor():
    args = ([0.006], [0], [0], [0.6], [1.2], [4.8])
    legacy, _ = retime_derivative_bounds(*args, mode="legacy_v1")
    new, bounds = retime_derivative_bounds(*args)
    assert legacy.end > 3.9 and new.end < 0.5
    assert np.all(bounds <= np.array([[0.6], [1.2], [4.8]]))
    np.testing.assert_allclose(new.sample(0)[0], [0])
    np.testing.assert_allclose(new.sample(new.end)[0], [1])


def test_random_cubic_bounds_survive_faster_duration():
    from emet.rl.manipulation.coupled_tracking import spline_bounds

    rng = np.random.default_rng(5)
    spline = CubicSpline([0, 0.3, 0.7, 1], rng.uniform(-0.2, 0.2, (4, 5)), bc_type="natural")
    path, bounds = retime_derivative_bounds(*spline_bounds(spline), np.full(5, 0.6), np.full(5, 1.2), np.full(5, 4.8))
    for t in np.linspace(0, path.end, 2001):
        q, v, a, j = (float(x[0]) for x in path.sample(t))
        first, second, third = (spline(q, k) for k in (1, 2, 3))
        observed = [first * v, second * v * v + first * a, third * v**3 + 3 * second * v * a + first * j]
        assert np.all(np.abs(observed) <= bounds * (1 + 1e-8))


def test_new_manifest_has_fresh_identities_and_explicit_quality():
    from emet.rl.manipulation.arm_splines import manifest, motion_manifest

    old, new = manifest(), motion_manifest()
    assert not {c["seed"] for c in old["cases"]} & {c["seed"] for c in new["cases"]}
    assert new["finish_after_hold"] and new["reference_timing"] == "bounded_v2"
    assert new["levels"][0]["amplitude_rad"] == 0.3
    assert new["spread_requirements"]["minimum_local_position_excursion_m"] == 0.05


@pytest.mark.sim
def test_short_reference_ends_after_real_quiet_hold_without_25s_padding():
    from emet.rl.manipulation.command_runtime import task_pose_teacher_action
    from emet.rl.manipulation.coupled_tracking import COARSE_SUITE, CoupledTrackingEnv

    case = {
        "id": "innate_mars:motion-timing:test",
        "seed": 530000001,
        "split": "train",
        "mode": "hand_free_base",
        "travel_m": 0.0,
        "turn_rad": 0.0,
        "arm_delta": [0, 0.006, -0.006, 0, 0.008],
        "base_mask": [False] * 3,
    }
    env = CoupledTrackingEnv(
        "innate_mars", [case], suite_path=COARSE_SUITE, timing="bounded_v2", finish_after_hold=True
    )
    try:
        env.reset()
        e = env.unwrapped
        end = e.control.task_trajectory.end
        assert end < 1 and e.config.horizon < 4
        for _ in range(150):
            _, _, done, truncated, info = env.step(task_pose_teacher_action(e.control))
            if done or truncated:
                break
        assert done and not truncated and info["primitive_success"]
        assert info["settled_seconds"] >= 2 and info["ended_after_quiet_hold"]
        assert not info["native_terminated"] and not info["native_truncated"]
        assert e.control.stopped
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.zeros(8))
    finally:
        env.close()


@pytest.mark.sim
def test_goal_handoff_finishes_only_after_successor_hold():
    from emet.rl.manipulation.command_runtime import task_pose_teacher_action
    from emet.rl.manipulation.goal_change_tracking import GoalChangeTrackingEnv

    case = {
        "id": "innate_mars:motion-timing:handoff",
        "seed": 530000002,
        "split": "train",
        "mode": "hand_free_base",
        "travel_m": 0.0,
        "turn_rad": 0.0,
        "arm_delta": [0, 0.006, -0.006, 0, 0.008],
        "base_mask": [False] * 3,
    }
    change = {"travel_m": 0.0, "turn_rad": 0.0, "arm_delta": [0, -0.006, 0.006, 0, -0.008], "dwell_s": 0.25}
    env = GoalChangeTrackingEnv(
        "innate_mars", [case], changes={case["id"]: change}, timing="bounded_v2", finish_after_hold=True
    )
    try:
        env.reset()
        e = env.unwrapped
        assert not env.may_finish_reference()
        for _ in range(160):
            _, _, done, truncated, info = env.step(task_pose_teacher_action(e.control))
            if done or truncated:
                break
        assert info["goal_change_delivered"] and info["primitive_success"]
        assert done and not truncated and e.steps * e.profile.dt >= env.switch_at + env.successor.end + 2 - 0.05
    finally:
        env.close()


@pytest.mark.parametrize("motion,expected", [(3, (3, 7, 0)), (None, (0, 0, 10)), (0, (0, 10, 0))])
def test_mixture_distinguishes_motion_quiet_and_missing_tick_data(motion, expected):
    import gymnasium as gym
    from gymnasium import spaces

    from emet.rl.manipulation.whole_body_env import ControlStepMixture

    class Toy(gym.Env):
        action_space = spaces.Box(-1, 1, (1,), np.float32)
        observation_space = spaces.Box(-1, 1, (1,), np.float32)

        def reset(self, **kwargs):
            return np.zeros(1, np.float32), {}

        def step(self, action):
            info = {"elapsed_control_ticks": 10}
            if motion is not None:
                info["reference_motion_ticks"] = motion
            return np.zeros(1, np.float32), 0.0, True, False, info

    env = ControlStepMixture({"one": Toy()})
    env.reset()
    env.step(np.zeros(1))
    row = env.ledger()["one"]
    assert (row["reference_motion_ticks"], row["reference_quiet_ticks"], row["reference_unknown_ticks"]) == expected
