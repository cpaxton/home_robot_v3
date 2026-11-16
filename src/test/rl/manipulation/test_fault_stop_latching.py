"""Repeated stops must brake against one target, including loaded fault resets."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from emet.controller.manipulation.native_scene import model_signature
from emet.controller.task.tamp.mars_sim_robot import MarsSimRobot
from emet.eval.mars_tamp_control import CANONICAL_SUITE, fixture
from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.safety import whole_body_stopped

pytestmark = pytest.mark.sim
ROOT = Path(__file__).resolve().parents[4]
CASES = ROOT / "paper/data/whole_body_control/mars_matched_reference_v1/stop-cases.json"


@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
def test_repeated_fault_stop_keeps_first_arm_target_and_cannot_authorize_release(robot):
    env = ManipulationEnv(ManipulationConfig(robot=robot, scenario="reach", randomization_scale=0))
    try:
        env.reset(seed=1800010)
        c = env.control
        c.grip_reference = c.grip_target = env.profile.gripper_closed
        c.stop()
        first_target, first_grip = c.target.copy(), c.grip_target
        # Feedback moves after the first stop; it must not move the hold target.
        for group in c.q:
            env.data.qpos[group] += 0.01 / len(group)
        env.data.qpos[c.grip_q] = env.profile.gripper_open
        c.set_reference(c.positions(), phase="release", grip=env.profile.gripper_open, release=True)
        c.stop()
        np.testing.assert_array_equal(c.target, first_target)
        assert c.grip_reference == c.grip_target == first_grip and not c.release_authorized
        assert c.stopped and np.isfinite(env.data.ctrl).all()
        np.testing.assert_array_equal(c.velocity, 0)
        np.testing.assert_array_equal(c.base_command, 0)
    finally:
        env.close()


def test_first_fault_during_release_keeps_finite_grip_command_if_jaw_feedback_is_invalid():
    env = ManipulationEnv(ManipulationConfig(robot="innate_mars", scenario="reach", randomization_scale=0))
    try:
        env.reset(seed=1800010)
        c = env.control
        c.set_reference(c.positions(), phase="release", release=True)
        finite_target = c.grip_target
        env.data.qpos[c.grip_q] = np.nan
        c.stop()
        assert c.stopped and not c.release_authorized
        assert c.grip_target == finite_target and c.grip_reference == finite_target
        assert np.isfinite(env.data.ctrl).all()
    finally:
        env.close()


@pytest.mark.parametrize("case_id", ["left:20g:0", "left:50g:0", "right:20g:0"])
def test_recorded_loaded_fault_reset_confirms_stop_without_target_rebasing(case_id):
    spec = next(c for c in json.loads(CASES.read_text())["cases"] if c["case_id"] == case_id)
    case = next(c for c in json.loads(CANONICAL_SUITE.read_text())["cases"] if c["id"] == case_id)
    env, admission = fixture(case)
    try:
        assert admission["accepted"] and model_signature(env.model) == spec["source_model_signature"]
        # This is a diagnostic pose/velocity reset, not full integration replay.
        state = spec["snapshot"]
        env.data.qpos[:] = state["qpos"]
        env.data.qvel[:] = state["qvel"]
        env.data.ctrl[:] = state["ctrl"]
        env.data.time = state["time"]
        mujoco.mj_forward(env.model, env.data)
        robot = MarsSimRobot(env)
        c = robot.servo
        c.set_reference(c.positions(), phase="carry", grip=spec["grip_reference"])
        # The saved first-stop control includes this already accumulated trim.
        trim = np.asarray(state["ctrl"])[c.act] - np.asarray(spec["joint_position"])
        c.tracking_integral[:] = trim / c.profile.tracking_trim_gain
        assert np.max(abs(trim)) <= c.profile.tracking_trim_limit + 1e-8
        initial = c.positions().copy()
        assert robot.stop_and_confirm(spec["reason"])
        assert whole_body_stopped(c) and robot.stop_confirmed
        assert robot.failure == spec["reason"] and c.stopped and robot.policy_calls == 0
        np.testing.assert_array_equal(c.target, initial)
        assert c.grip_target == spec["grip_reference"] and not c.release_authorized
    finally:
        env.close()


def test_loaded_stop_reset_bytes_are_bound_to_the_source_regression_receipt():
    from emet.rl.regression_gate import file_digest, source_state

    assert source_state(ROOT)[str(CASES.relative_to(ROOT))] == file_digest(CASES)
