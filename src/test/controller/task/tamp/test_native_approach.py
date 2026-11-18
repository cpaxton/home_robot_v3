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


def test_loaded_sweep_checks_payload_only_obstacle_and_preserves_attachment():
    from emet.eval.physical_tamp import make_scene_checker
    m, d = scene(obstacle='''<body name="payload" pos="0 .12 .2"><freejoint/>
      <geom type="box" size=".004 .004 .004"/></body>
      <geom type="box" pos=".02 .12 .2" size=".002 .005 .005"/>''')
    checker = make_scene_checker(m, {'object_body': 'payload', 'gripper_bodies': ()})
    checker.set_payload(m, d, 'payload', 'base_link')
    original = d.qpos.copy()
    attachment = checker.payload_transform.copy()
    assert module.sweep_clear(m, d, [0, 0, 0], [.04, 0, 0], envelope=False)
    assert not module.sweep_clear(m, d, [0, 0, 0], [.04, 0, 0], envelope=False, loaded_checker=checker)
    np.testing.assert_array_equal(d.qpos, original)
    np.testing.assert_array_equal(checker.payload_transform, attachment)


@pytest.mark.parametrize('when', ['before_command', 'during_motion'])
def test_loaded_motion_checks_retention_after_preflight_and_stops_on_loss(monkeypatch, when):
    monkeypatch.setenv('EMET_SIM_NAV_TELEPORT', '0')
    m, d = scene()
    clock = [0.]
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock[0],
        sleep=lambda dt: clock.__setitem__(0, clock[0]+dt)))
    monkeypatch.setattr(module, 'sweep_clear', lambda *args, **kwargs: True)
    robot = SimpleNamespace(get_emet_session=lambda: {'is_simulation': True},
        move_base_to=Mock(return_value=True), cancel_navigation=Mock(return_value=True))
    def sync(data):
        data.time = clock[0]
        mujoco.mj_kinematics(m, data)
    retained = Mock(side_effect=[True, False] if when=='before_command' else [True, True, False])
    ex = SimpleNamespace(model=m, data=d, synchronize=sync, payload_body='arm', _loaded_binding_done=True,
        collision=SimpleNamespace(payload_body='arm'), payload_retained=retained, qadr=np.array([7]),
        joint_tolerance=.03, event=Mock())
    result = module.execute_approach(robot, ex, module.BaseApproach((0.,0.,0.),(.04,0.,0.)), loaded=True)
    assert result.reason == 'payload_not_retained'
    assert result.execution_started is (when=='during_motion')
    assert robot.move_base_to.called is (when=='during_motion')
    assert robot.cancel_navigation.called is (when=='during_motion')


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
    robot.cancel_navigation.assert_called_once()


def test_quaternion_sign_is_not_scene_motion_or_mutation():
    from emet.controller.manipulation.native_scene import qpos_difference
    model, data = scene(obstacle='<body name="object" pos="1 0 1"><freejoint/><geom size=".01"/></body>')
    original = data.qpos.copy()
    flipped = original.copy()
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            a = int(model.jnt_qposadr[j])+3
            flipped[a:a+4] *= -1
    np.testing.assert_array_equal(qpos_difference(model, flipped, original), 0.)
    changed = flipped.copy()
    changed[0] += .02
    assert qpos_difference(model, changed, original)[0] == pytest.approx(.02)
    np.testing.assert_array_equal(data.qpos, original)


@pytest.mark.parametrize('drift', ['base', 'arm', 'stalled_clock', None])
def test_grasp_requires_fresh_quiet_base_and_arm(monkeypatch, drift):
    model, data = scene()
    clock = [0.]
    def sync(d):
        d.time = 0. if drift == 'stalled_clock' else clock[0]
        d.qpos[0] = .04 + (.02 if drift == 'base' else 0.)
        d.qpos[7] = .005*np.sin(clock[0]*30) if drift == 'arm' else 0.
        mujoco.mj_kinematics(model, d)
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock[0],
        sleep=lambda dt: clock.__setitem__(0, clock[0]+dt)))
    ex = SimpleNamespace(model=model, data=data, qadr=np.array([7]), synchronize=sync)
    assert module.verify_approach_ready(ex, module.BaseApproach((0., 0., 0.), (.04, 0., 0.))) is (drift is None)


def test_changed_geometry_rejects_before_dispatch(monkeypatch):
    monkeypatch.setenv('EMET_SIM_NAV_TELEPORT', '0')
    m, d = scene(obstacle='<geom type="box" pos=".02 0 .1" size=".002 .01 .01"/>')
    robot = SimpleNamespace(get_emet_session=lambda: {'is_simulation': True},
        move_base_to=Mock(), cancel_navigation=Mock())
    executor = SimpleNamespace(model=m, data=d, synchronize=lambda data: None, event=Mock())
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


def test_arrival_budget_starts_after_expensive_preflight(monkeypatch):
    monkeypatch.setenv('EMET_SIM_NAV_TELEPORT', '0')
    m, d = scene()
    clock, sent, checked = [0.], [False], [False]
    def sweep(*args, **kwargs):
        if not checked[0]:
            checked[0] = True
            clock[0] += 20.
        return True
    def send(*args, **kwargs):
        sent[0] = True
        return True
    def sync(data):
        data.time = clock[0]
        if sent[0]:
            data.qpos[0] = min(.04, .04*(clock[0]-20.))
        mujoco.mj_kinematics(m, data)
    monkeypatch.setattr(module, 'sweep_clear', sweep)
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock[0],
        sleep=lambda dt: clock.__setitem__(0, clock[0]+dt)))
    robot = SimpleNamespace(get_emet_session=lambda: {'is_simulation': True},
        move_base_to=Mock(side_effect=send), cancel_navigation=Mock(return_value=True))
    ex = SimpleNamespace(model=m, data=d, synchronize=sync, qadr=np.array([7]),
                         joint_tolerance=.03, event=Mock())
    result = module.execute_approach(robot, ex, module.BaseApproach((0., 0., 0.), (.04, 0., 0.)))
    assert result.success and clock[0] >= 21.5
