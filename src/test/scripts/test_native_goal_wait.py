# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from emet.eval.native_goal_wait import wait_for_native_goal


@pytest.mark.parametrize('behavior', ['biased', 'blocked', 'accurate', 'transit', 'uncompensated', 'wrong_pose'])
def test_native_arrival_corrects_bias_but_does_not_accept_blocked_motion(monkeypatch, behavior):
    nested = '<body name="tool"><geom type="sphere" size=".01" mass=".1"/></body>'
    for i in reversed(range(8)):
        kind, axis = ('slide', '0 0 1') if i == 7 else ('hinge', '0 0 1')
        nested = (f'<body pos=".1 0 0"><joint name="j{i}" type="{kind}" axis="{axis}" '
                  f'range="-1 1" limited="true"/><geom type="sphere" size=".01" mass=".1"/>{nested}</body>')
    model = mujoco.MjModel.from_xml_string('<mujoco><compiler angle="radian"/><worldbody>'+nested+'</worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    clock = [0.]
    monkeypatch.setattr('emet.eval.native_goal_wait.time', SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda dt: clock.__setitem__(0, clock[0]+dt)))
    commands = []
    goal = np.array([.2, 0., 0., 0., 0., 0., 0., .1])

    def synchronize(data):
        if behavior != 'blocked':
            data.qpos[:] = commands[-1]
            if behavior in {'biased', 'transit', 'uncompensated'}:
                data.qpos[0] -= .05
        mujoco.mj_kinematics(model, data)

    arm = SimpleNamespace(chain=SimpleNamespace(ee_body='tool'),
        command_joints=lambda q: commands.append(q.copy()), synchronize=synchronize)
    limits = np.tile([-1., 1.], (8, 1))
    options = {}
    if behavior in {'transit', 'uncompensated'}:
        options['max_corrections'] = 0
    if behavior == 'transit':
        options.update(position_tolerance_m=.04, rotation_tolerance_rad=.2)
    if behavior == 'wrong_pose':
        options.update(require_task_pose=True, task_position=np.array([100., 0., 0.]))
    result = wait_for_native_goal(arm, model, data, np.arange(8), limits, goal, **options)
    assert result['arrived'] is (behavior not in {'blocked', 'uncompensated', 'wrong_pose'})
    if behavior in {'accurate', 'transit', 'uncompensated'}:
        assert len(commands) == 1 and result['corrections'] == []
    else:
        assert 1 < len(commands) <= 4
    assert clock[0] <= 4.03
    for command in commands:
        assert np.all(np.abs(command-goal) <= np.r_[np.full(7, .120001), .004001])
        assert np.all((command >= limits[:, 0]) & (command <= limits[:, 1]))
    if behavior == 'biased':
        assert abs(data.qpos[0]-goal[0]) < .015
    if behavior == 'blocked':
        assert result['tool_position_error_m'] > .008
