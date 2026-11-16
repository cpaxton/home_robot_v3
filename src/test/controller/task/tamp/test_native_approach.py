# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Private full-body sweeps and actual measured approach boundaries."""
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest

from emet.controller.task.tamp import native_approach as module


def scene(*, obstacle='', arm=False):
    m = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="base_link" pos="0 0 .1"><freejoint/>
        <geom type="box" size=".004 .004 .004"/>
        <body name="arm" pos="0 .15 .1"><joint name="arm_joint"/>
          <geom type="box" size=".006 .006 .006" contype="{int(arm)}" conaffinity="{int(arm)}"/>
        </body>
      </body>{obstacle}</worldbody></mujoco>''')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_clear_endpoints_do_not_authorize_colliding_base_interior():
    m, d = scene(obstacle='<geom type="box" pos=".02 0 .1" size=".002 .01 .01"/>')
    before = {n: getattr(d, n).copy() for n in ('qpos', 'qvel', 'qacc', 'xpos')}
    assert module.sweep_clear(m, d, [0, 0, 0], [0, 0, 0], envelope=False)
    assert module.sweep_clear(m, d, [.04, 0, 0], [.04, 0, 0], envelope=False)
    assert not module.sweep_clear(m, d, [0, 0, 0], [.04, 0, 0], envelope=False)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(d, name), value)


def test_sweep_includes_arm_and_controller_tracking_envelope():
    m, d = scene(arm=True, obstacle='<geom type="box" pos=".02 .15 .2" size=".002 .01 .01"/>')
    assert not module.sweep_clear(m, d, [0, 0, 0], [.04, 0, 0], envelope=False)
    m, d = scene(obstacle='<geom type="box" pos=".02 .012 .1" size=".002 .002 .01"/>')
    assert module.sweep_clear(m, d, [0, 0, 0], [.04, 0, 0], envelope=False)
    assert not module.sweep_clear(m, d, [0, 0, 0], [.04, 0, 0])


@pytest.mark.parametrize('fault', [None, 'ack_only', 'false_ack', 'timeout', 'bad_stop', 'stale', 'corridor'])
def test_approach_requires_measured_arrival_and_confirmed_stop(monkeypatch, fault):
    monkeypatch.setenv('EMET_SIM_NAV_TELEPORT', '0')
    m, d = scene()
    clock, dispatched = [0.], [False]
    def send(*args, **kwargs):
        dispatched[0] = True
        if fault == 'timeout':
            raise TimeoutError('may have been sent')
        return fault != 'false_ack'
    robot = SimpleNamespace(get_emet_session=lambda: {'is_simulation': True},
        move_base_to=Mock(side_effect=send), cancel_navigation=Mock(return_value=fault != 'bad_stop'))
    def sync(data):
        if dispatched[0] and fault == 'stale':
            raise RuntimeError('stale_observation')
        data.time = clock[0]
        if dispatched[0] and fault != 'ack_only':
            data.qpos[0] = min(.04, .04 * clock[0])
            if fault == 'corridor':
                data.qpos[1] = .02
        mujoco.mj_kinematics(m, data)
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock[0],
        sleep=lambda dt: clock.__setitem__(0, clock[0]+dt)))
    executor = SimpleNamespace(model=m, data=d, synchronize=sync, qadr=np.array([7]),
                               joint_tolerance=.03, event=Mock())
    result = module.execute_approach(robot, executor, module.BaseApproach((0., 0., 0.), (.04, 0., 0.)))
    assert result.success is (fault is None)
    assert result.execution_started is True
    assert robot.cancel_navigation.called
    if fault is None:
        assert clock[0] >= 1.5  # Arrival plus separate quiet intervals before/after cancellation.
        assert executor.event.call_args.kwargs['phase'] == 'base_approach_complete'
    elif fault == 'bad_stop':
        assert result.reason == 'navigation_cancellation_unconfirmed'
    elif fault == 'timeout':
        assert result.reason == 'command_outcome_unknown'


def test_changed_geometry_rejects_before_dispatch(monkeypatch):
    monkeypatch.setenv('EMET_SIM_NAV_TELEPORT', '0')
    m, d = scene(obstacle='<geom type="box" pos=".02 0 .1" size=".002 .01 .01"/>')
    robot = SimpleNamespace(get_emet_session=lambda: {'is_simulation': True},
        move_base_to=Mock(), cancel_navigation=Mock())
    executor = SimpleNamespace(model=m, data=d, synchronize=lambda data: None)
    result = module.execute_approach(robot, executor, module.BaseApproach((0., 0., 0.), (.04, 0., 0.)))
    assert result.reason == 'base_approach_invalidated' and not result.execution_started
    robot.move_base_to.assert_not_called()
    robot.cancel_navigation.assert_not_called()


def test_teleport_capability_cannot_enter_physical_approach(monkeypatch):
    monkeypatch.setenv('EMET_SIM_NAV_TELEPORT', '0')
    robot = SimpleNamespace(get_emet_session=lambda: {'is_simulation': True, 'capabilities': {'teleport_base': True}},
        move_base_to=Mock(), cancel_navigation=Mock())
    assert not module.approach_available(robot)


def test_moving_object_revalidates_remaining_sweep_before_arrival(monkeypatch):
    monkeypatch.setenv('EMET_SIM_NAV_TELEPORT', '0')
    m, d = scene(obstacle='''<body name="moving" pos=".5 0 .1"><freejoint/>
      <geom type="box" size=".002 .004 .01"/></body>''')
    clock, sent = [0.], [False]
    def send(*args, **kwargs):
        sent[0] = True
        return True
    robot = SimpleNamespace(get_emet_session=lambda: {'is_simulation': True},
        move_base_to=Mock(side_effect=send), cancel_navigation=Mock(return_value=True))
    address = int(m.jnt_qposadr[m.body('moving').jntadr[0]])
    def sync(data):
        data.time = clock[0]
        if sent[0]:
            data.qpos[0] = min(.01, .01*clock[0])
            data.qpos[address] = .025
        mujoco.mj_kinematics(m, data)
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock[0],
        sleep=lambda dt: clock.__setitem__(0, clock[0]+dt)))
    executor = SimpleNamespace(model=m, data=d, synchronize=sync, qadr=np.array([7]),
                               joint_tolerance=.03, event=Mock())
    result = module.execute_approach(robot, executor, module.BaseApproach((0., 0., 0.), (.04, 0., 0.)))
    assert not result.success and result.execution_started
    assert clock[0] < .5
    assert executor.event.call_args.kwargs['reason'] == 'scene_changed_collision'
    robot.cancel_navigation.assert_called_once()
