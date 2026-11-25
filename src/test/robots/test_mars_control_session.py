"""Loopback-only supervised whole-body driver boundary; never connect hardware."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

from emet.rl.manipulation.profiles import MARS
from emet.robots.innate_mars import control_session
from emet.robots.innate_mars.control_session import GuardedControlSession


class Driver:
    def __init__(self):
        self.pose = [0.0, 0.0, 0.0]
        self.age = 0.0
        self.writes, self.stops = [], 0
        self.confirmed = True
        self.measured_velocity = np.zeros(MARS.arm_size)
        self.base_velocity = np.zeros(6)
        self.arm_age = 0.0
        self.valid = True
        self.caps = {
            "whole_body_velocity": True,
            "measured_whole_body_stop": True,
            "robot_watchdog_s": 0.2,
            "feedback_si_units_verified": True,
            "motor_read_validity_verified": True,
        }

    def capabilities(self):
        return self.caps

    def feedback(self):
        return {
            "base_pose": self.pose,
            "arm_captured_monotonic": time.monotonic() - self.arm_age,
            "odom_captured_monotonic": time.monotonic() - self.age,
            "valid": self.valid,
            "base_twist": self.base_velocity.copy(),
            "joint_velocity": self.measured_velocity.copy(),
        }

    def command_velocity(self, value, *, valid_for_s):
        self.writes.append((value, valid_for_s))

    def stop_and_confirm(self, *, timeout_s):
        self.stops += 1
        assert timeout_s <= 0.75
        return self.confirmed


def test_disabled_or_incomplete_driver_never_writes():
    d = Driver()
    with pytest.raises(RuntimeError, match="disabled"):
        GuardedControlSession(driver=d, profile=MARS).start()
    d.caps["whole_body_velocity"] = False
    with pytest.raises(RuntimeError, match="capabilities"):
        GuardedControlSession(driver=d, profile=MARS, enabled=True).start()
    assert not d.writes


@pytest.mark.parametrize("stage", ["feedback", "age"])
def test_session_start_preserves_abort_before_guard_creation(monkeypatch, stage):
    driver = Driver()
    session = GuardedControlSession(driver=driver, profile=MARS, enabled=True)
    original = SystemExit(2)

    def abort(*args):
        raise original

    monkeypatch.setattr(
        driver if stage == "feedback" else session, "feedback" if stage == "feedback" else "_age", abort
    )
    with pytest.raises(SystemExit) as raised:
        session.start()
    assert raised.value is original and session.guard is None
    assert session.fault == "session start aborted" and not driver.writes
    with pytest.raises(RuntimeError, match="restarted"):
        session.start()


def test_profile_acceleration_and_target_limits_apply_before_dispatch():
    d = Driver()
    s = GuardedControlSession(driver=d, profile=MARS, enabled=True)
    s.start()
    s.last_command_time -= 0.05
    s.command(np.ones(7) * 0.1, base_target=[0, 0, 0])
    assert len(d.writes) == 1
    assert d.writes[0][0][0] <= MARS.base_acceleration * 0.06
    with pytest.raises(RuntimeError, match="target outside"):
        s.command(np.zeros(7), base_target=[0.001, 0, 0])
    assert len(d.writes) == 1 and d.stops == 1


def test_wbc_action_uses_direct_base_and_arm_si_velocities():
    d = Driver()
    d.pose = [1.0, -2.0, 0.0]
    session = GuardedControlSession(driver=d, profile=MARS, enabled=True, translation_limit_m=0.12)
    session.start()
    session.last_command_time -= 0.05
    action = [0.02, -0.02, 0.02, 0.0, -0.02, 0.0, 0.0, 1.0]
    assert session.command_action(action, base_target=[1.1, -2.0, 0.0])
    velocity, lifetime = d.writes[0]
    np.testing.assert_allclose(velocity, [0.0024, -0.01, 0.012, 0, -0.012, 0, 0])
    assert lifetime == 0.2

    d.pose = [1.1, -2.0, 0.0]
    with pytest.raises(RuntimeError, match="target outside"):
        session.command_action(action, base_target=[1.13, -2.0, 0.0])
    assert d.stops == 1 and len(d.writes) == 1


def test_direct_translation_stops_at_the_measured_boundary():
    d = Driver()
    session = GuardedControlSession(
        driver=d, profile=MARS, enabled=True, translation_limit_m=0.1, translation_tolerance_m=0.02
    )
    session.start()
    d.pose = [0.121, 0, 0]
    with pytest.raises(SystemExit):
        session.command_action(np.zeros(8), base_target=[0.08, 0, 0])
    assert session.fault and d.stops == 1 and not d.writes


def test_invalid_wbc_action_latches_stop_before_publishing():
    d = Driver()
    session = GuardedControlSession(driver=d, profile=MARS, enabled=True)
    session.start()
    with pytest.raises(RuntimeError, match="invalid normalized WBC action"):
        session.command_action([0.0] * 7 + [float("nan")], base_target=[0, 0, 0])
    assert session.fault and d.stops == 1 and not d.writes


@pytest.mark.sim
def test_direct_wbc_moves_arm_and_base_in_the_same_schedule(monkeypatch):
    from emet.rl.manipulation.command_runtime import task_pose_teacher_action
    from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
    from emet.rl.manipulation.safety import whole_body_stopped
    from emet.rl.manipulation.tracking import TaskFrameFK, TaskPoseTrajectory

    config = ManipulationConfig(
        robot="innate_mars",
        scenario="reach",
        action_mode="direct_velocity",
        trajectory_preview=True,
        randomization_scale=0,
        horizon=12.0,
    )
    env = ManipulationEnv(config, mesh_storage="binary")
    now = 100.0
    monkeypatch.setattr(control_session, "time", SimpleNamespace(monotonic=lambda: now))
    limits = np.r_[MARS.base_speed, MARS.base_yaw_speed, MARS.speed]
    control = env.control

    class SimulatorDriver(Driver):
        def feedback(self):
            first = control.base_dof
            return {
                "base_pose": control.base_pose(),
                "arm_captured_monotonic": now,
                "odom_captured_monotonic": now,
                "valid": True,
                "base_twist": env.data.qvel[first : first + 6].copy(),
                "joint_velocity": control.velocities(),
            }

        def command_velocity(self, value, *, valid_for_s):
            super().command_velocity(value, valid_for_s=valid_for_s)
            self.result = env.step(np.r_[value / limits, 0.0])

    try:
        _, initial = env.reset(seed=1900000)
        assert not initial["initialization_failed"]
        driver = SimulatorDriver()
        session = GuardedControlSession(
            driver=driver,
            profile=MARS,
            enabled=True,
            translation_limit_m=0.10,
            translation_tolerance_m=0.02,
        )
        session.start()
        base_start = control.base_pose()
        arm_start = control.positions()
        heading = np.array([np.cos(base_start[2]), np.sin(base_start[2]), 0.0])
        base_end = base_start + 0.08 * heading
        arm_end = arm_start.copy()
        arm_end[2] += 0.075
        fk = TaskFrameFK(env.model, env.data, control)
        path = TaskPoseTrajectory(
            [0, 5],
            [base_start, base_end],
            [control.ee_pose(), fk(base_end, arm_end)],
            [control.grip_reference] * 2,
            base_mask=[True] * 3,
        )
        env.set_task_trajectory(path)
        for _ in range(round(config.horizon / MARS.dt)):
            now += MARS.dt
            action = task_pose_teacher_action(control)
            assert session.command_action(action, base_target=path.base[-1])
            _, _, terminated, truncated, info = driver.result
            if terminated or truncated:
                break

        assert info["is_success"], info
        assert whole_body_stopped(control)
        assert (control.base_pose() - base_start) @ heading > 0.07
        assert control.positions()[2] - arm_start[2] > 0.06
        assert any(abs(v[0]) > 0.001 and abs(v[4]) > 0.01 for v, _ in driver.writes)
    finally:
        env.close()


@pytest.mark.parametrize("pose,age", [([0.051, 0, 0], 0), ([0, 0, np.deg2rad(48)], 0), ([0, 0, 0], 0.201)])
def test_measured_boundary_stops_and_exits_before_velocity_dispatch(pose, age):
    d = Driver()
    s = GuardedControlSession(driver=d, profile=MARS, enabled=True)
    s.start()
    d.pose, d.age = pose, age
    with pytest.raises(SystemExit) as exc:
        s.command(np.zeros(7), base_target=[0, 0, 0])
    assert exc.value.code == 2
    assert d.stops == 1 and not d.writes
    with pytest.raises(RuntimeError, match="not active"):
        s.command(np.zeros(7), base_target=[0, 0, 0])


def test_target_loss_and_command_gap_latch_stop_without_automatic_resume():
    d = Driver()
    s = GuardedControlSession(driver=d, profile=MARS, enabled=True)
    s.start()
    assert s.command(np.zeros(7), base_target=[0, 0, 0], target_visible=False)
    assert not d.writes and d.stops == 1
    with pytest.raises(RuntimeError, match="restarted"):
        s.start()
    s = GuardedControlSession(driver=d, profile=MARS, enabled=True)
    s.start()
    s.last_command_time -= 0.201
    with pytest.raises(RuntimeError, match="stream interrupted"):
        s.command(np.zeros(7), base_target=[0, 0, 0])
    assert not d.writes


def test_unconfirmed_stop_is_reported_and_fault_retained():
    d = Driver()
    s = GuardedControlSession(driver=d, profile=MARS, enabled=True)
    s.start()
    d.confirmed = False
    with pytest.raises(RuntimeError, match="stop_confirmed=False"):
        s.command(np.ones(7) * 10, base_target=[0, 0, 0])
    assert s.fault and not d.writes


@pytest.mark.parametrize("velocity", [np.full(MARS.arm_size, 0.03), np.full(MARS.arm_size, np.nan), np.zeros(4)])
def test_nonstationary_or_invalid_initial_arm_feedback_stops_and_latches(velocity):
    d = Driver()
    d.measured_velocity = velocity
    session = GuardedControlSession(driver=d, profile=MARS, enabled=True)
    with pytest.raises(RuntimeError, match="stationary|invalid arm feedback"):
        session.start()
    assert session.fault and d.stops == 1 and not d.writes
    with pytest.raises(RuntimeError, match="restarted"):
        session.start()


def test_measured_overspeed_rejects_even_a_zero_command_before_dispatch():
    d = Driver()
    session = GuardedControlSession(driver=d, profile=MARS, enabled=True)
    session.start()
    d.measured_velocity[2] = MARS.speed[2] + 0.01
    with pytest.raises(RuntimeError, match="measured arm velocity exceeds"):
        session.command(np.zeros(7), base_target=[0, 0, 0])
    assert session.fault and d.stops == 1 and not d.writes


def test_in_range_feedback_is_accepted_without_rebasing_command_slew():
    d = Driver()
    session = GuardedControlSession(driver=d, profile=MARS, enabled=True)
    session.start()
    session.last_command_time -= 0.05
    d.measured_velocity[:] = 0.2
    session.command(np.ones(7) * 0.1, base_target=[0, 0, 0])
    assert len(d.writes) == 1 and d.stops == 0
    assert np.max(d.writes[0][0][2:]) < 0.1


@pytest.mark.parametrize("component", range(6))
def test_base_velocity_all_six_components_checked_before_start_and_dispatch(component):
    d = Driver()
    d.base_velocity[component] = 0.1
    with pytest.raises(RuntimeError, match="base must be stationary"):
        GuardedControlSession(driver=d, profile=MARS, enabled=True).start()
    assert not d.writes and d.stops == 1
    d = Driver()
    session = GuardedControlSession(driver=d, profile=MARS, enabled=True)
    session.start()
    d.base_velocity[component] = 0.6
    with pytest.raises(RuntimeError, match="measured base velocity"):
        session.command(np.zeros(7), base_target=[0, 0, 0])
    assert not d.writes and d.stops == 1


@pytest.mark.parametrize("key", ["feedback_si_units_verified", "motor_read_validity_verified"])
def test_unverified_feedback_cannot_activate_session(key):
    d = Driver()
    d.caps[key] = False
    with pytest.raises(RuntimeError, match="capabilities required"):
        GuardedControlSession(driver=d, profile=MARS, enabled=True).start()
    assert not d.writes


def test_fresh_odometry_cannot_hide_stale_arm_or_invalid_read():
    d = Driver()
    d.arm_age = 0.3
    session = GuardedControlSession(driver=d, profile=MARS, enabled=True)
    with pytest.raises(RuntimeError, match="stale arm"):
        session.start()
    assert session.fault and d.stops == 1 and not d.writes
    d = Driver()
    session = GuardedControlSession(driver=d, profile=MARS, enabled=True)
    session.start()
    d.valid = False
    with pytest.raises(RuntimeError, match="valid motor"):
        session.command(np.zeros(7), base_target=[0, 0, 0])
    assert d.stops == 1 and not d.writes


def test_adaptive_elbow_load_slows_joint_and_stops_on_fault(monkeypatch):
    driver = Driver()
    feedback = driver.feedback
    load = 32.0

    def measured():
        return feedback() | {"elbow_load_percent": load, "arm_healthy": True}

    monkeypatch.setattr(driver, "feedback", measured)
    session = GuardedControlSession(driver=driver, profile=MARS, enabled=True, adaptive_elbow_load=True)
    with pytest.raises(RuntimeError, match="verified XL430"):
        session.start()
    driver.caps["elbow_load_percent_verified"] = True
    session.start()

    load = 37.5
    session.last_command_time -= 0.1
    requested = np.array([0, 0, 0, 0.02, -0.04, 0, 0])
    assert session.command(requested, base_target=[0, 0, 0])
    np.testing.assert_allclose(driver.writes[-1][0], [0, 0, 0, 0.02, -0.02, 0, 0])
    assert session.elbow_speed_scale == 0.5

    load = 40.0
    with pytest.raises(RuntimeError, match="immediate ceiling.*stop_confirmed=True"):
        session.command(requested, base_target=[0, 0, 0])
    assert len(driver.writes) == driver.stops == 1
