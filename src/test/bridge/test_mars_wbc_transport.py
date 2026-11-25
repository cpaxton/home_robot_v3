import numpy as np
import pytest
from innate_mars_bridge.wbc_transport import MarsWatchdogTransport


class Loopback:
    now = 10.0
    exclusive = True
    verified = True
    stale = False

    def __init__(self):
        self.arm, self.base = [], []
        self.transport = MarsWatchdogTransport(
            feedback=self.feedback,
            owns_topics=lambda: self.exclusive,
            publish_arm=self.arm.append,
            publish_base=self.base.append,
            clock=lambda: self.now,
        )

    def feedback(self):
        stamp = self.now - (1 if self.stale else 0.001)
        return {
            "valid": True,
            "arm_healthy": True,
            "feedback_si_units_verified": self.verified,
            "motor_read_validity_verified": self.verified,
            "arm_captured_monotonic": stamp,
            "odom_captured_monotonic": stamp,
            "joint_position": [0.2] * 5,
            "joint_velocity": [0.0] * 5,
            "base_twist": [0.0] * 6,
            "gripper_target_rad": 0.3,
        }

    def send(self):
        self.transport.tick()
        self.transport.send_targets(arm_position_rad=[0.25] * 6, base_twist_si=[0.03, 0.01], valid_for_s=0.1)


def test_watchdog_expires_and_confirms_measured_hold():
    robot = Loopback()
    transport = robot.transport
    assert not transport.capabilities()["watchdog_arm_position_and_base_twist"]
    transport.tick()
    assert transport.capabilities()["watchdog_arm_position_and_base_twist"]
    robot.send()
    robot.now += 0.11
    transport.tick()
    np.testing.assert_allclose(robot.base[-1], [0, 0])
    np.testing.assert_allclose(robot.arm[-1], [0.2] * 5 + [0.3])
    assert not transport._stop_confirmed

    robot.now += 0.02
    transport.tick()
    robot.now += 0.16
    transport.tick()
    assert transport.stop_and_confirm()
    with pytest.raises(RuntimeError, match="new ownership session"):
        robot.send()


def test_lost_ownership_or_stale_feedback_sends_no_new_arm_target():
    for reason in ("ownership", "stale"):
        robot = Loopback()
        robot.send()
        if reason == "ownership":
            robot.exclusive = False
        else:
            robot.stale = True
        robot.now += 0.02
        robot.transport.tick()
        assert len(robot.arm) == 1
        np.testing.assert_allclose(robot.base[-1], [0, 0])
        assert not robot.transport._stop_confirmed


def test_unverified_feedback_cannot_start_motion():
    robot = Loopback()
    robot.verified = False
    with pytest.raises(RuntimeError, match="verified feedback lost"):
        robot.send()
    assert not robot.arm


def test_new_stream_requires_fresh_measured_stop():
    robot = Loopback()
    transport = robot.transport
    robot.send()
    robot.now += 0.11
    transport.tick()
    assert not transport.can_rearm()
    with pytest.raises(RuntimeError, match="confirmed stop"):
        transport.rearm()
    robot.now += 0.02
    transport.tick()
    robot.now += 0.16
    transport.tick()
    assert transport.can_rearm()
    transport.rearm()
    robot.send()
    np.testing.assert_allclose(robot.base[-1], [0.03, 0.01])
