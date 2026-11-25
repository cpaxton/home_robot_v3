import math
import time

import numpy as np
import pytest

from emet.rl.manipulation.profiles import MARS
from emet.robots.innate_mars.startup_recovery import JOINT_NAMES
from emet.robots.innate_mars.wbc_interface import MarsWBCInterface

LIMITS = dict(
    zip(
        JOINT_NAMES,
        [(-1.57, 1.571), (-1.22, 1.57), (-1.75, 1.57), (-1.75, 1.92), (-1.57, 1.57), (0, 0.87)],
        strict=True,
    )
)


class Transport:
    ready = True

    def __init__(self):
        self.joints = np.zeros(5)
        self.targets = []
        self.stops = 0

    def capabilities(self):
        return {
            "watchdog_arm_position_and_base_twist": self.ready,
            "measured_whole_body_stop": True,
            "feedback_si_units_verified": True,
            "motor_read_validity_verified": True,
            "robot_watchdog_s": 0.2,
        }

    def feedback(self):
        stamp = time.monotonic()
        return {
            "valid": True,
            "arm_captured_monotonic": stamp,
            "odom_captured_monotonic": stamp,
            "base_pose": [0, 0, 0],
            "base_twist": np.zeros(6),
            "joint_velocity": np.zeros(5),
            "joint_position": self.joints,
            "gripper_target_rad": 0.4,
        }

    def send_targets(self, **packet):
        self.targets.append(packet)

    def stop_and_confirm(self, *, timeout_s):
        self.stops += 1
        return True


def interface(transport):
    return MarsWBCInterface(
        transport=transport,
        profile=MARS,
        published_limits=LIMITS,
        policy=lambda observation: observation,
        enabled=True,
    )


def test_policy_steps_stream_measured_relative_joints_and_wheel_twist():
    t = Transport()
    wbc = interface(t)
    wbc.start()
    action = [0.02, -0.02, 0.02, 0.0, -0.02, 0.0, 0.0, 1.0]

    for i in range(2):
        t.joints[0] = i * 0.01
        wbc.session.last_command_time -= 0.1
        assert wbc.step(action, base_target=[0, 0, 0])
        packet = t.targets[-1]
        np.testing.assert_allclose(packet["base_twist_si"], [0.0024, -0.01])
        np.testing.assert_allclose(
            packet["arm_position_rad"], np.r_[t.joints + MARS.dt * np.array([0.012, 0, -0.012, 0, 0]), 0.4]
        )
        assert packet["valid_for_s"] == 0.2

    wbc.step(action, base_target=[0, 0, 0], target_visible=False)
    assert t.stops == 1
    assert len(t.targets) == 2


def test_adapter_defers_yaw_at_innate_shoulder_limit():
    t = Transport()
    t.joints[:2] = [1.125, -0.86]
    wbc = interface(t)
    wbc.start()
    wbc.session.last_command_time -= 0.1
    wbc.step([0, 0, -0.5, 0, 0, 0, 0, 0], base_target=[0, 0, 0])
    assert math.isclose(t.targets[0]["arm_position_rad"][0], 1.125)
    assert math.isclose(t.targets[0]["arm_position_rad"][1], -0.86)


def test_stock_capabilities_cannot_authorize_wbc():
    t = Transport()
    t.ready = False
    wbc = interface(t)
    with pytest.raises(RuntimeError, match="capabilities"):
        wbc.start()
    assert not t.targets


def test_robot_side_executor_samples_waypoint_from_measured_feedback():
    import mujoco
    from scipy.spatial.transform import Rotation

    from emet.core.pose_stream import pose_stream_goal
    from emet.rl.manipulation.control import ManipulationControl
    from emet.rl.manipulation.models import build_manipulation_model
    from emet.rl.manipulation.whole_body_observation import pose_rotation
    from emet.robots.innate_mars.wbc_runtime import MarsPoseExecutor

    class MeasuredTransport(Transport):
        def feedback(self):
            return super().feedback() | {"arm_healthy": True, "gripper_state": [0.06, 0.0], "gripper_target_rad": 0.06}

    transport = MeasuredTransport()
    model, _ = build_manipulation_model("innate_mars")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    control = ManipulationControl(model, data, MARS)
    transport.joints[:] = control.positions()
    executor = MarsPoseExecutor(transport=transport, control=control, published_limits=LIMITS)
    hand = control.ee_pose()
    goal = pose_stream_goal(
        {"arms": {"arm": [*hand[:3], *Rotation.from_matrix(pose_rotation(hand)).as_quat()]}},
        duration_s=0.12,
        frame="odom",
        free_base=False,
        capabilities=executor.stream.capabilities(),
    )
    executor.start(goal)
    from emet.core.pose_stream import FORMAT

    executor.update({"pose_stream": {"format": FORMAT, "finish": True}})
    try:
        assert executor._done.wait(2.0)
        assert executor.result()[0] == "succeeded", executor.result()
        assert transport.targets
        assert executor.stop()
    finally:
        if not executor._done.is_set():
            executor.stop()
