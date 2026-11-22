"""Exercise the Mars arm transport contract without ROS or hardware."""

import runpy
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

arm = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "innate_mars_bridge/innate_mars_bridge/remote/arm_motion.py")
)
MarsArmMotion = arm["MarsArmMotion"]


@pytest.mark.parametrize("goal", [{"base": [0, 0, 0]}, {"arm": {"joint6": 0.4}}, {"arm": {"joint1": float("nan")}}])
def test_invalid_arm_goals(goal):
    with pytest.raises(ValueError):
        MarsArmMotion.validate(goal)


def test_position_ramp_requires_driver_echo_and_measured_settling(monkeypatch):
    clock = Mock(return_value=0.0)
    monkeypatch.setattr(arm["time"], "monotonic", clock)
    ros = Mock()
    ros.get_published_arm_limits.return_value = arm["JOINT_LIMITS"]
    measured = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.25])
    ros.arm_motion_feedback.side_effect = lambda: (measured.copy(), np.zeros(6))
    ros.publish_arm_targets.side_effect = lambda command: np.copyto(measured, command)
    ros.arm_target_applied.return_value = False
    motion = MarsArmMotion(ros)
    motion.start({"arm": {"joint6": 0.15}})
    for step in range(1, 30):
        clock.return_value = step * 0.05
        previous = measured.copy()
        assert motion.poll() is None
        assert np.array_equal(measured[:5], previous[:5])
        assert abs(measured[5] - previous[5]) <= 0.00750001
    assert measured[5] == 0.15
    ros.arm_target_applied.return_value = True
    clock.return_value = 1.5
    assert motion.poll() is None
    clock.return_value = 1.8
    assert motion.poll()[0] == "succeeded"


def test_conflicting_arm_publisher_blocks_motion():
    ros = Mock()
    ros.get_published_arm_limits.return_value = arm["JOINT_LIMITS"]
    ros.verify_arm_command_path.side_effect = RuntimeError("arm owned elsewhere")
    motion = MarsArmMotion(ros)
    with pytest.raises(RuntimeError, match="owned elsewhere"):
        motion.start({"arm": {"joint6": 0.15}})
    assert motion.stop()
    ros.publish_arm_targets.assert_not_called()


def test_startup_check_uses_named_goal_limits_and_rejects_partial_feedback():
    names = list(arm["JOINT_LIMITS"])
    assert arm["startup_joint_positions"](names[:-1], [0.0] * 5) is None
    assert arm["startup_joint_positions"](names, [float("nan")] * 6) is None
    measured = [0.0, -1.6398, 0.0, 0.0, 0.0, 0.0]
    positions = arm["startup_joint_positions"](names[::-1], measured[::-1])
    assert arm["outside_joint_limits"](positions) == ["joint2"]
    ros = Mock()
    ros.get_published_arm_limits.return_value = arm["JOINT_LIMITS"]
    ros.arm_motion_feedback.return_value = np.array(measured), np.zeros(6)
    with pytest.raises(RuntimeError, match="Measured Mars joint outside published limits: joint2"):
        MarsArmMotion(ros).start({"arm": {"joint5": 0.1}})
    ros.publish_arm_targets.assert_not_called()


def setup_recovery():
    ros = Mock()
    ros.get_published_arm_limits.return_value = arm["JOINT_LIMITS"]
    start = np.array([1.50, -1.64, 1.095, 0.94, 0.00, 0.26])
    ros.arm_motion_feedback.return_value = start.copy(), np.zeros(6)
    driver = Mock()
    driver.ready.return_value = True
    controller = arm["MarsJointRecovery"](ros, driver)
    goal = {
        "arm_recovery": {
            "joint": "joint2",
            "target_rad": -1.53,
            "start_arm_rad": dict(zip(arm["JOINT_LIMITS"], start.tolist(), strict=True)),
        },
    }
    return controller, goal, start, ros, driver


def test_recovery_dispatches_only_selected_joint_and_requires_measured_in_range():
    controller, goal, start, ros, driver = setup_recovery()
    controller.start(goal)
    joint, initial, target, duration = driver.start.call_args.args
    assert (joint, initial, target) == ("joint2", -1.64, -1.53)
    assert duration == pytest.approx(0.11 / arm["MAX_SPEED"])
    driver.start.return_value.outcome.return_value = "succeeded"
    assert controller.poll()[0] == "failed"
    start[1] = target
    ros.arm_motion_feedback.return_value = start, np.zeros(6)
    assert controller.poll()[0] == "succeeded"
    ros.publish_arm_targets.assert_not_called()
    ros.verify_arm_command_path.assert_not_called()


def test_recovery_refuses_a_live_ordinary_command_receiver():
    controller, goal, _, ros, driver = setup_recovery()
    ros.verify_arm_recovery_path.side_effect = RuntimeError("ordinary Mars arm command receiver")
    with pytest.raises(RuntimeError, match="ordinary Mars arm command receiver"):
        controller.start(goal)
    driver.start.assert_not_called()


def test_native_recovery_requires_measured_settling_after_service_success(monkeypatch):
    _, goal, current, ros, driver = setup_recovery()
    motion = arm["MarsJointRecovery"](ros, driver, native=True)
    motion.start(goal)
    driver.start.return_value.outcome.return_value = "succeeded"
    assert motion.poll() is None
    driver.start.assert_called_once()
    assert driver.start.call_args.kwargs["start_arm"] == current.tolist()

    current[1] = -1.53
    ros.arm_motion_feedback.return_value = current.copy(), np.zeros(6)
    clock = Mock(return_value=0.0)
    monkeypatch.setattr(arm["time"], "monotonic", clock)
    assert motion.poll() is None
    clock.return_value = 0.26
    assert motion.poll()[0] == "succeeded"

    current[1] = -1.642
    ros.arm_motion_feedback.return_value = current, np.zeros(6)
    with pytest.raises(RuntimeError, match="farther beyond"):
        motion.poll()


def test_native_recovery_can_stage_from_a_reviewed_in_range_pose():
    _, goal, current, ros, driver = setup_recovery()
    current[1] = -1.56
    ros.arm_motion_feedback.return_value = current, np.zeros(6)
    goal["arm_recovery"]["start_arm_rad"]["joint2"] = -1.56
    motion = arm["MarsJointRecovery"](ros, driver, native=True)
    with pytest.raises(RuntimeError, match="already inside"):
        motion.start(goal)
    goal["arm_recovery"]["stage_only"] = True
    motion.start(goal)
    assert driver.start.call_args.args[:3] == ("joint2", -1.56, -1.53)


def test_recovery_rejects_missing_driver_and_stale_start():
    controller, goal, _, ros, driver = setup_recovery()
    driver.ready.return_value = False
    with pytest.raises(RuntimeError, match="no isolated"):
        controller.start(goal)
    driver.ready.return_value = True
    goal["arm_recovery"]["start_arm_rad"]["joint3"] += 0.1
    with pytest.raises(RuntimeError, match="moved since"):
        controller.start(goal)
    driver.start.assert_not_called()
    ros.publish_arm_targets.assert_not_called()


def test_recovery_rejects_in_range_joint_even_if_another_joint_is_outside():
    controller, goal, _, _, driver = setup_recovery()
    goal["arm_recovery"].update(joint="joint3", target_rad=1.0)
    with pytest.raises(RuntimeError, match="already inside"):
        controller.start(goal)
    driver.start.assert_not_called()


def test_recovery_faults_on_other_joint_motion_and_confirms_cancel(monkeypatch):
    controller, goal, current, ros, driver = setup_recovery()
    controller.start(goal)
    current[2] += 0.03
    ros.arm_motion_feedback.return_value = current, np.zeros(6)
    with pytest.raises(RuntimeError, match="unselected"):
        controller.poll()
    pending = driver.start.return_value
    pending.outcome.return_value = "cancelled"
    times = iter(np.arange(0, 1, 0.03))
    monkeypatch.setattr(arm["time"], "monotonic", lambda: next(times))
    monkeypatch.setattr(arm["time"], "sleep", lambda _: None)
    assert controller.stop()
    pending.cancel.assert_called_once()
    ros.publish_arm_targets.assert_not_called()


def test_cancellation_while_waiting_for_isolated_driver_acceptance():
    from concurrent.futures import Future

    recovery = runpy.run_path(str(Path(arm["__file__"]).with_name("joint_recovery.py")))
    response = Future()
    handle = Mock(accepted=True)
    handle.get_result_async.return_value = Future()
    goal = recovery["RecoveryGoal"](response)
    goal.cancel()
    handle.cancel_goal_async.assert_not_called()
    response.set_result(handle)
    goal.cancel()
    handle.cancel_goal_async.assert_called_once()


def test_recovery_rejects_a_driver_clamp_past_the_requested_interval():
    controller, goal, current, ros, driver = setup_recovery()
    controller.start(goal)
    current[1] = -1.22
    ros.arm_motion_feedback.return_value = current, np.zeros(6)
    with pytest.raises(RuntimeError, match="left the isolated recovery interval"):
        controller.poll()
    ros.publish_arm_targets.assert_not_called()


def test_recovery_staging_requires_an_explicit_request_and_an_outside_joint():
    controller, goal, current, ros, driver = setup_recovery()
    goal["arm_recovery"].update(joint="joint3", target_rad=1.0, stage_only=True)
    controller.start(goal)
    assert driver.start.call_args.args[:3] == ("joint3", 1.095, 1.0)
    driver.start.reset_mock()
    current[1] = -1.53
    goal["arm_recovery"]["start_arm_rad"]["joint2"] = float(current[1])
    ros.arm_motion_feedback.return_value = current, np.zeros(6)
    with pytest.raises(RuntimeError, match="requires an in-range joint and another joint outside"):
        controller.start(goal)
    driver.start.assert_not_called()


def test_recovery_driver_unconfirmed_stop_cannot_become_a_success():
    controller, goal, _, _, driver = setup_recovery()
    controller.start(goal)
    driver.start.return_value.outcome.return_value = "stop_unconfirmed"
    assert not controller.stop()


def test_named_goal_uses_limits_from_live_robot_description():
    limits = arm["JOINT_LIMITS"] | {"joint3": (-1.7453, 1.5708)}
    description = (
        "<robot>"
        + "".join(
            f'<joint name="{name}"><limit lower="{lower}" upper="{upper}"/></joint>'
            for name, (lower, upper) in limits.items()
        )
        + "</robot>"
    )
    ros = Mock()
    ros.get_published_arm_limits.return_value = arm["published_arm_limits"](description)
    assert ros.get_published_arm_limits.return_value == limits
    ros.arm_motion_feedback.return_value = np.array([0.0, -1.218, 1.57233, 0.0, 0.0, 0.15]), np.zeros(6)
    with pytest.raises(RuntimeError, match="outside published limits: joint3"):
        MarsArmMotion(ros).start({"arm": {"joint5": 0.1}})
    ros.publish_arm_targets.assert_not_called()


@pytest.mark.parametrize("fresh_joint3", [1025 * arm["ENCODER_STEP_RAD"], 1024 * arm["ENCODER_STEP_RAD"]])
def test_recovery_uses_published_080_range_and_allows_one_count_feedback_change(fresh_joint3):
    controller, goal, current, ros, driver = setup_recovery()
    limits = arm["JOINT_LIMITS"] | {"joint3": (-1.7453, 1.5708)}
    ros.get_published_arm_limits.return_value = limits
    current[2] = fresh_joint3
    reviewed = current.tolist()
    reviewed[2] = 1025 * arm["ENCODER_STEP_RAD"]
    ros.arm_motion_feedback.return_value = current, np.zeros(6)
    goal["arm_recovery"].update(
        joint="joint3", target_rad=1.50, start_arm_rad=dict(zip(limits, reviewed, strict=True))
    )
    controller.start(goal)
    assert driver.start.call_args.args[:3] == ("joint3", fresh_joint3, 1.50)
