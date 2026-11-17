"""The rehearsal wave holds base/grip and has bounded, continuous FK targets."""

from types import SimpleNamespace

import numpy as np
import pytest

from emet.rl.manipulation.env import ManipulationConfig
from emet.rl.manipulation.herman_gestures import HermanWaveReference, wave_case
from emet.rl.manipulation.herman_rehearsal_env import HermanRehearsalEnv
from emet.rl.manipulation.jerk_reference import polynomial_peaks
from emet.rl.manipulation.models import build_manipulation_model


def test_wave_is_offline_and_keeps_all_base_coordinates_active():
    case = wave_case()
    assert case["base_mask"] == [True] * 3 and not case["hardware_executable"]
    case["base_mask"][0] = False
    assert wave_case()["base_mask"] == [True] * 3
    with pytest.raises(ValueError, match="requires Mars"):
        HermanWaveReference(SimpleNamespace(profile=SimpleNamespace(robot="stretch")))


@pytest.mark.sim
def test_wave_preserves_history_state_base_and_gripper_and_obeys_derivative_limits():
    env = HermanRehearsalEnv(
        ManipulationConfig(robot="innate_mars", scenario="reach", randomization_scale=0),
        initial_arm=[1.5308, -1.5308, 1.0631264228554684, 1.0906603401866908, -0.019941750242513337],
        initial_grip=0.08897088569736719,
        initial_head=0,
        model_factory=build_manipulation_model,
    )
    try:
        env.reset(seed=430_010_000)
        c = env.control
        qpos, qvel, time = env.data.qpos.copy(), env.data.qvel.copy(), float(env.data.time)
        initial = c.positions().copy()
        reference = HermanWaveReference(c)
        assert reference.end < 23  # Leaves at least two seconds to settle in the 25s rehearsal.
        assert reference.initial_hold_s == 2
        assert reference.base_mask.tolist() == [True] * 3 and reference.hand_mask
        np.testing.assert_array_equal(env.data.qpos, qpos)
        np.testing.assert_array_equal(env.data.qvel, qvel)
        assert env.data.time == time
        for part in reference.path.segments:
            for actual, limit in zip(polynomial_peaks(part), [0.03, 0.06, 0.24], strict=True):
                assert np.all(actual <= limit * (1 + 1e-8))
            for t in [0, part.duration]:
                # Each reversal is rest-to-rest with continuous velocity/acceleration.
                from numpy.polynomial.polynomial import polyder, polyval

                for order in [1, 2]:
                    np.testing.assert_allclose(polyval(t, polyder(part.coefficients, order, axis=0)), 0, atol=1e-8)
        samples = [reference.sample(t) for t in np.linspace(0, reference.end + 1, 101)]
        for base, hand, grip in samples:
            np.testing.assert_allclose(base, c.base_pose(), atol=1e-9)
            assert grip == c.grip_reference and np.isfinite(hand).all()
        np.testing.assert_allclose(reference.coordinates(1)[1], initial)
        np.testing.assert_allclose(reference.coordinates(reference.end + 10)[1], initial, atol=1e-10)
        np.testing.assert_allclose(samples[0][1], samples[-1][1], atol=1e-9)
    finally:
        env.close()


def test_endpoint_return_cannot_hide_failure_to_follow_the_wave():
    from emet.rl.manipulation.herman_gestures import wave_tracking_gate

    trace = [
        {"time_s": t, "position_error_m": 0, "rotation_error_rad": rotation}
        for t, rotation in [(2, 0), (5.75, 0.06), (9.5, 0), (13.25, 0.06), (17, 0), (25, 0)]
    ]
    assert not wave_tracking_gate(trace, 17)["passed"]
    followed = [{**sample, "rotation_error_rad": 0.001} for sample in trace]
    assert wave_tracking_gate(followed, 17)["passed"]
    assert not wave_tracking_gate(followed[:3], 17)["passed"]
    assert not wave_tracking_gate([], 17)["passed"]
