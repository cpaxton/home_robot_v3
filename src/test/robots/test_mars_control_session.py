"""Loopback-only supervised whole-body driver boundary; never connect hardware."""

import time

import numpy as np
import pytest

from emet.rl.manipulation.profiles import MARS
from emet.robots.innate_mars.control_session import GuardedControlSession


class Driver:
    def __init__(self):
        self.pose = [0., 0., 0.]
        self.age = 0.
        self.writes, self.stops = [], 0
        self.confirmed = True
        self.caps = {"whole_body_velocity": True, "measured_whole_body_stop": True, "robot_watchdog_s": 0.2}

    def capabilities(self):
        return self.caps

    def feedback(self):
        return {"base_pose": self.pose, "captured_monotonic": time.monotonic() - self.age,
                "joint_velocity": np.zeros(MARS.arm_size)}

    def command_velocity(self, value, *, valid_for_s):
        self.writes.append((value, valid_for_s))

    def stop_and_confirm(self, *, timeout_s):
        self.stops += 1
        assert timeout_s <= 0.75
        return self.confirmed


def test_disabled_or_incomplete_driver_never_writes():
    d = Driver()
    with pytest.raises(RuntimeError, match='disabled'):
        GuardedControlSession(driver=d, profile=MARS).start()
    d.caps['whole_body_velocity'] = False
    with pytest.raises(RuntimeError, match='capabilities'):
        GuardedControlSession(driver=d, profile=MARS, enabled=True).start()
    assert not d.writes


def test_profile_acceleration_and_target_limits_apply_before_dispatch():
    d = Driver()
    s = GuardedControlSession(driver=d, profile=MARS, enabled=True)
    s.start()
    s.last_command_time -= 0.05
    s.command(np.ones(7) * .1, base_target=[0, 0, 0])
    assert len(d.writes) == 1
    assert d.writes[0][0][0] <= MARS.base_acceleration * .06
    with pytest.raises(RuntimeError, match='target outside'):
        s.command(np.zeros(7), base_target=[.001, 0, 0])
    assert len(d.writes) == 1 and d.stops == 1


@pytest.mark.parametrize('pose,age', [([.051,0,0],0), ([0,0,np.deg2rad(48)],0), ([0,0,0],.201)])
def test_measured_boundary_stops_and_exits_before_velocity_dispatch(pose, age):
    d = Driver()
    s = GuardedControlSession(driver=d, profile=MARS, enabled=True)
    s.start()
    d.pose, d.age = pose, age
    with pytest.raises(SystemExit) as exc:
        s.command(np.zeros(7), base_target=[0,0,0])
    assert exc.value.code == 2
    assert d.stops == 1 and not d.writes
    with pytest.raises(RuntimeError, match='not active'):
        s.command(np.zeros(7), base_target=[0,0,0])


def test_target_loss_and_command_gap_latch_stop_without_automatic_resume():
    d = Driver()
    s = GuardedControlSession(driver=d, profile=MARS, enabled=True)
    s.start()
    assert s.command(np.zeros(7), base_target=[0,0,0], target_visible=False)
    assert not d.writes and d.stops == 1
    with pytest.raises(RuntimeError, match='restarted'):
        s.start()
    s = GuardedControlSession(driver=d, profile=MARS, enabled=True)
    s.start()
    s.last_command_time -= .201
    with pytest.raises(RuntimeError, match='stream interrupted'):
        s.command(np.zeros(7), base_target=[0,0,0])
    assert not d.writes


def test_unconfirmed_stop_is_reported_and_fault_retained():
    d = Driver()
    s = GuardedControlSession(driver=d, profile=MARS, enabled=True)
    s.start()
    d.confirmed = False
    with pytest.raises(RuntimeError, match='stop_confirmed=False'):
        s.command(np.ones(7) * 10, base_target=[0,0,0])
    assert s.fault and not d.writes
