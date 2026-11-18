"""Loopback-only supervised whole-body driver boundary; never connect hardware."""

import time

import numpy as np
import pytest

from emet.rl.manipulation.profiles import MARS
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
