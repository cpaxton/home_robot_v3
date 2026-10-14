import numpy as np
import pytest

from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.oracle import oracle_robot_class, stretch_arm_reference
from emet.rl.manipulation.tamp_cases import execute_case, plan_case

pytestmark = pytest.mark.sim


def test_public_arm_mapping_rejects_base_motion():
    np.testing.assert_array_equal(stretch_arm_reference([0, 1, 2, 3, 4, 5]), [1, 2, 5, 4, 3])
    with pytest.raises(ValueError, match="base motion"):
        stretch_arm_reference([0.1, 1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        stretch_arm_reference([0, 1, 2, np.nan, 4, 5])


def test_real_tamp_planner_uses_private_state_and_produces_coupled_path():
    env = ManipulationEnv(ManipulationConfig(scenario="reach"))
    try:
        env.reset(seed=600001)
        before = env.data.qpos.copy()
        target = env.control.positions() + [0.025, 0, 0, 0, 0]
        result = plan_case(env, target)
        np.testing.assert_array_equal(env.data.qpos, before)
        assert result["planning_error"] is None, result
        assert len(result["waypoints"]) >= 2
        names = result["joint_names"]
        extensions = [names.index(f"joint_arm_l{i}") for i in range(4)]
        path = np.asarray(result["waypoints"])
        # Planner endpoints preserve each measured telescope coordinate; the
        # public command sums them, while the goal shares extension equally.
        np.testing.assert_allclose(path[-1, extensions], target[1] / 4)
        np.testing.assert_allclose(path[-1], result["goal"])
        assert not result["execution_verified"]
        execution = execute_case(env, result)
        assert execution["execution_verified"], execution
        assert execution["settled_hold_seconds"] >= 1
    finally:
        env.close()


def test_oracle_command_is_actuator_only_and_stop_cannot_release_payload():
    env = ManipulationEnv(ManipulationConfig(scenario="reach"))
    try:
        env.reset(seed=600002)

        class Base:
            def __init__(self, model, data):
                self.model, self.data = model, data
                self.commands = []

        robot = oracle_robot_class(Base)(env.model, env.data)
        q = env.control.positions()
        before = robot.data.qpos.copy()
        assert robot.arm_to([0, q[0] + 0.02, q[1], q[4], q[3], q[2]], head=[0, 0], blocking=False)
        np.testing.assert_array_equal(robot.data.qpos, before)
        assert robot.close_gripper(blocking=False)
        robot.step(0.1)
        assert robot.servo.grip_reference == env.profile.gripper_closed
        assert robot.open_gripper(blocking=False)
        robot.phase = "pregrasp"
        assert robot.arm_to([0, q[0], q[1], q[4], q[3], q[2]], head=[0, 0], blocking=False)
        assert robot.servo.grip_reference == env.profile.gripper_open
        assert robot.close_gripper(blocking=False)
        target = robot.servo.grip_target
        robot.failure = "stale_observation"
        robot.servo.stop()
        assert not robot.open_gripper(blocking=True)
        assert robot.servo.grip_target == target
        robot.data.qpos[robot.servo.q[0]] = np.nan
        robot.servo.stop()
        assert np.isfinite(robot.data.ctrl).all()
    finally:
        env.close()
