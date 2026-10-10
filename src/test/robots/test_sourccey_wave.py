# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Sourccey differential kinematics, IK, and emote interface regressions."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from emet.core.task import Task
from emet.motion.mujoco_arm_ik import joint_dof_addrs, joint_qpos_addrs, solve_position_ik
from emet.robots.sourccey import SourcceyBackend
from emet.robots.sourccey.emote_backend import SourcceyWaveOperation, wave_trajectory


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(SourcceyBackend().get_spec().mjcf_path)


def home(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("sourccey_home").id)
    mujoco.mj_forward(model, data)
    return data


@pytest.mark.parametrize("side", ["left", "right"])
def test_jacobian_and_fk_ik_roundtrip(model, side):
    chain = SourcceyBackend().get_spec().arm_chains[side]
    qa = joint_qpos_addrs(model, chain.joint_names)
    da = joint_dof_addrs(model, chain.joint_names)
    rng = np.random.default_rng(19)
    for _ in range(12):
        data = home(model)
        data.qpos[:4] = [0.7, -0.4, 0.6, -0.12]
        data.qpos[qa] = np.asarray(chain.home_arm_q) * 0.7 + rng.uniform(-0.15, 0.15, 5)
        mujoco.mj_forward(model, data)
        target = data.body(chain.ee_body).xpos.copy()
        jac = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jac, None, model.body(chain.ee_body).id)
        finite = []
        for address in qa:
            original = data.qpos[address]
            data.qpos[address] = original + 1e-6
            mujoco.mj_forward(model, data)
            plus = data.body(chain.ee_body).xpos.copy()
            data.qpos[address] = original - 1e-6
            mujoco.mj_forward(model, data)
            minus = data.body(chain.ee_body).xpos.copy()
            finite.append((plus - minus) / 2e-6)
            data.qpos[address] = original
        np.testing.assert_allclose(jac[:, da], np.array(finite).T, atol=1e-7)
        data.qpos[qa] += rng.uniform(-0.08, 0.08, 5)
        before = data.qpos.copy()
        result = solve_position_ik(
            model,
            data,
            ee_body=chain.ee_body,
            joint_names=chain.joint_names,
            target_pos=target,
            tol_m=0.001,
            max_iters=150,
        )
        assert result.success and result.pos_error_m < 0.001
        other = [i for i in range(model.nq) if i not in qa]
        np.testing.assert_array_equal(data.qpos[other], before[other])
    result = solve_position_ik(
        model, data, ee_body=chain.ee_body, joint_names=chain.joint_names, target_pos=[10, 10, 10]
    )
    assert not result.success


@pytest.mark.parametrize("side", ["left", "right"])
def test_wave_limits_speed_and_other_actuators(model, side):
    q = home(model).qpos.copy()
    q[:3] = [2, -1, 1.2]  # Never send these positions to the velocity actuators.
    frames = wave_trajectory(q, side)
    np.testing.assert_array_equal(frames[:, :3], 0)
    assert np.max(np.abs(np.diff(frames[:, 3:], axis=0))) / 0.05 <= 0.80001
    assert np.all(frames[:, 3:] >= model.actuator_ctrlrange[3:, 0] - 1e-9)
    assert np.all(frames[:, 3:] <= model.actuator_ctrlrange[3:, 1] + 1e-9)
    np.testing.assert_allclose(frames[-1, 3:], q[3:], atol=1e-9)
    selected = [
        SourcceyBackend().get_spec().joint_names.index(n)
        for n in SourcceyBackend().get_spec().arm_chains[side].joint_names
    ]
    held = [i for i in range(3, 16) if i not in selected]
    np.testing.assert_allclose(frames[:, held], np.broadcast_to(q[held], frames[:, held].shape))


class FeedbackRobot:
    def __init__(self, q, *, stuck=False, at_goal=True):
        self.q = q.copy()
        self.stuck = stuck
        self.idle = at_goal
        self.commands = []

    def at_goal(self):
        return self.idle

    def get_joint_state(self, timeout=1):
        return self.q.copy(), None, None

    def switch_to_manipulation_mode(self):
        pass

    def set_actuator_positions(self, target):
        self.commands.append(target.copy())
        if not self.stuck:
            self.q[3:] = target[3:]


@pytest.mark.parametrize("side", ["left", "right"])
def test_backend_wave_runs_through_task(model, monkeypatch, side):
    monkeypatch.setattr("emet.robots.sourccey.emote_backend.time.sleep", lambda _: None)
    robot = FeedbackRobot(home(model).qpos)
    task = Task()
    SourcceyBackend().get_emote_backend().add_named_emote(task, f"wave_{side}", SimpleNamespace(robot=robot))
    assert task.run()
    assert len(robot.commands) > 100


@pytest.mark.parametrize("failure", ["stuck", "busy", "invalid", "missing", "dropout", "connection", "base_drift"])
def test_wave_failure_is_reported(model, monkeypatch, failure):
    monkeypatch.setattr("emet.robots.sourccey.emote_backend.time.sleep", lambda _: None)
    robot = FeedbackRobot(home(model).qpos, stuck=failure == "stuck", at_goal=failure != "busy")
    if failure == "invalid":
        robot.q[4] = np.nan
    if failure == "missing":
        robot.get_joint_state = lambda **_: (None, None, None)
    if failure == "dropout":
        robot.get_joint_state = lambda **_: (None, None, None) if robot.commands else (robot.q, None, None)
    if failure == "connection":

        def disconnected(*args, **kwargs):
            raise ConnectionError("Connection lost")

        robot.set_actuator_positions = disconnected
    if failure == "base_drift":
        original = robot.set_actuator_positions

        def drifting(target):
            original(target)
            robot.q[0] += 0.1

        robot.set_actuator_positions = drifting
    operation = SourcceyWaveOperation("wave", SimpleNamespace(robot=robot))
    operation.run()
    assert not operation.was_successful()
    assert operation.error
    if failure not in ("stuck", "dropout", "base_drift"):
        assert not robot.commands
